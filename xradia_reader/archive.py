"""Compressed HDF5 archive of Xradia files.

Only stock HDF5 filters are used (byte *shuffle* + *gzip/deflate*), which every
libhdf5 build since the 1990s decodes -- h5py, MATLAB ``h5read``, Fiji HDF5
import, HDFView, Avizo, Dragonfly... No plugins are needed to read the files.

Layout (format "xradia-archive", version 3)::

    /data            stored dtype (uint16 for Versa reconstructions), shape
                     (n, h, w), one chunk per image, shuffle + gzip
        attrs: element_size_um [z, y, x], unit, axes, drop_bits,
               quantization_step, noise_sigma_estimate, pixel_sha256,
               source_pixel_sha256, and ImageJ calibration (pixelWidth,
               pixelHeight, pixelDepth, pixelUnit="um", numSlices...)
    /reference       flat-field reference (projection files only)
    /summary         empty group; its attributes hold the key values
                     (voxel size, kV, uA, filter, objective, distances...)
    /metadata        uint8: UTF-8 JSON (gzip filter) with "summary" (the full
                     curated summary) and "tree" (every decoded stream)
    /raw_streams     uint8: UTF-8 JSON (gzip filter), {stream path: base64 of
                     the raw bytes} for every non-pixel OLE stream
    / attrs          format, format_version, source_file, source_size,
                     source_sha256, created, kind, writer

In the JSON, arrays are ``{"__ndarray__": dtype, "shape": [...], "base64":
raw little-endian bytes}`` and byte strings ``{"__bytes__": base64}``; read
them with ``json.loads(bytes(f["metadata"][()]))``.  Metadata is packed into
two datasets so that Fiji's HDF5 import lists only a handful of entries.

Older archives are still read; :func:`upgrade_archive` converts them without
recompressing the pixels (version 1: one dataset per metadata stream;
version 2: no ImageJ attributes).

``drop_bits=k`` (opt-in) rounds integer values to the nearest multiple of 2**k
before compression ("near-lossless"); zeros (outside the reconstructed field of
view) stay zero.  With k chosen well below the noise level the change is
statistically invisible but compression improves a lot (see README).
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from .reader import ImageStack, XradiaFile, _get as _get_path, _is_pixel_stream, _json_default, key_values

FORMAT = "xradia-archive"
FORMAT_VERSION = 3

#: ImagePlus type codes used by the ImageJ attributes on /data.
_IJ_TYPE = {np.dtype("u1"): 0, np.dtype("<u2"): 1, np.dtype("<f4"): 2}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def quantize(img: np.ndarray, k: int) -> np.ndarray:
    """Round unsigned-integer ``img`` to the nearest multiple of 2**k.

    Zeros stay zero; the result is clipped to the dtype maximum.
    Maximum absolute change is 2**(k-1).
    """
    if k <= 0:
        return img
    if img.dtype.kind != "u":
        raise TypeError("quantize() needs unsigned integer data")
    top = np.iinfo(img.dtype).max
    q = ((img.astype(np.uint32) + (1 << (k - 1))) >> k) << k
    q = np.minimum(q, top).astype(img.dtype)
    q[img == 0] = 0
    return q


def estimate_noise(stack: ImageStack, n_samples: int = 10) -> float | None:
    """Robust per-voxel noise sigma (MAD of horizontal neighbour differences,
    inside the non-zero field of view) -- the minimum over ``n_samples``
    images spread through the stack."""
    n = stack.n_images
    sig = []
    for i in np.linspace(0, n - 1, min(n, n_samples)).astype(int):
        x = stack.image(int(i)).astype(np.float64)
        m = (x[:, 1:] != 0) & (x[:, :-1] != 0)
        if m.sum() < 1000:
            continue
        d = np.diff(x, axis=1)[m]
        sig.append(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    return float(min(sig)) if sig else None


def auto_drop_bits(sigma: float | None, safety: float = 16.0, max_bits: int = 8) -> int:
    """Largest k with 2**k <= sigma / safety (0 if unknown), capped."""
    if not sigma or sigma <= safety:
        return 0
    return int(min(max_bits, np.floor(np.log2(sigma / safety))))


def _shuffle(b: bytes, itemsize: int) -> bytes:
    """HDF5 shuffle filter: byte-transpose elements of ``itemsize`` bytes."""
    if itemsize == 1:
        return b
    a = np.frombuffer(b, np.uint8).reshape(-1, itemsize)
    return np.ascontiguousarray(a.T).tobytes()


def _sha256_file(path, bufsize=16 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(bufsize):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# metadata <-> HDF5
# ---------------------------------------------------------------------------
def _json_enc(o):
    if isinstance(o, np.ndarray):
        return {"__ndarray__": o.dtype.str, "shape": list(o.shape),
                "base64": base64.b64encode(np.ascontiguousarray(o).tobytes()).decode()}
    if isinstance(o, bytes):
        return {"__bytes__": base64.b64encode(o).decode()}
    return _json_default(o)


def _json_hook(d):
    if "__ndarray__" in d:
        return np.frombuffer(base64.b64decode(d["base64"]), np.dtype(d["__ndarray__"])) \
            .reshape(d["shape"]).copy()
    if "__bytes__" in d:
        return base64.b64decode(d["__bytes__"])
    return d


def _put_json(f: h5py.File, name: str, obj, **attrs):
    b = json.dumps(obj, default=_json_enc, separators=(",", ":")).encode()
    ds = f.create_dataset(name, data=np.frombuffer(b, np.uint8),
                          chunks=(min(len(b), 1 << 20),) if b else None,
                          compression="gzip" if b else None, compression_opts=9 if b else None)
    ds.attrs.update({"content": "UTF-8 JSON", **attrs})


def _get_json(ds: h5py.Dataset):
    return json.loads(ds[()].tobytes(), object_hook=_json_hook)


def _write_key_values(f: h5py.File, summary: dict):
    g = f.create_group("summary", track_order=True)
    for k, v in key_values(summary).items():
        g.attrs[k] = v if isinstance(v, (int, float, bool, np.number)) else str(v)


def _write_imagej_attrs(d: h5py.Dataset, title: str):
    """ImageJ calibration attributes on /data, read by Fiji's
    File > Import > HDF5/N5/Zarr/OME-NGFF (n5-ij "ImageJ" metadata), which
    then shows the unit as um.  The older File > Import > HDF5 (HDF5_Vibez)
    reads only element_size_um and always labels it "micrometer"."""
    dz, dy, dx = (float(v) for v in d.attrs["element_size_um"])
    d.attrs.update({
        "title": title, "pixelWidth": dx, "pixelHeight": dy, "pixelDepth": dz,
        "pixelUnit": "um", "xOrigin": 0.0, "yOrigin": 0.0, "zOrigin": 0.0,
        "numChannels": 1, "numSlices": int(d.shape[0]), "numFrames": 1,
        "fps": 0.0, "frameInterval": 0.0, "type": _IJ_TYPE.get(d.dtype, 1),
    })


# version-1 layout: one dataset per stream
def _get(ds: h5py.Dataset):
    t = ds.attrs.get("py_type")
    if t == "none":
        return None
    if t == "bytes":
        return ds[()].tobytes()
    if t == "list":
        return list(ds.asstr()[()])
    if t == "bool":
        return bool(ds[()])
    if ds.dtype.kind == "O" or h5py.check_string_dtype(ds.dtype):
        return ds.asstr()[()]
    v = ds[()]
    if isinstance(v, np.ndarray):
        return v
    return v.item()


def _read_tree(group: h5py.Group) -> dict:
    out = {}
    for k, v in group.items():
        out[k] = _read_tree(v) if isinstance(v, h5py.Group) else _get(v)
    return out


# ---------------------------------------------------------------------------
# writer
# ---------------------------------------------------------------------------
def write_hdf5(xf: XradiaFile, path, drop_bits: int | str = 0, level: int = 4,
               workers: int | None = None, verify: bool = True,
               source_hash: bool = True, progress=None, sel=None) -> dict:
    """Archive an open :class:`XradiaFile` to ``path`` (HDF5).

    ``drop_bits``: 0 (lossless, default), an int k, or ``"auto"`` (k from the
    measured noise, see :func:`auto_drop_bits`).  Integer data only.
    ``sel`` (a slice) archives only some images (mainly for testing).
    Returns a report dict (sizes, ratio, timings, verification result).
    """
    t0 = time.time()
    path = Path(path)
    idx = range(xf.n_images) if sel is None else range(*sel.indices(xf.n_images))
    n, (h, w), dt = len(idx), xf.image_shape, xf.dtype
    itemsize = dt.itemsize
    workers = workers or os.cpu_count() or 1

    sigma = estimate_noise(xf) if dt.kind in "ui" else None
    if drop_bits == "auto":
        drop_bits = auto_drop_bits(sigma)
    drop_bits = int(drop_bits)
    if drop_bits and dt.kind != "u":
        raise ValueError(f"drop_bits needs unsigned integer data, file has {dt}")

    px = xf.pixel_size_um or 1.0
    src_hash = hashlib.sha256()
    out_hash = hashlib.sha256()
    max_err = 0

    def compress(img):
        q = quantize(img, drop_bits)
        err = int(np.abs(q.astype(np.int64) - img).max()) if drop_bits else 0
        return q, err, zlib.compress(_shuffle(q.tobytes(), itemsize), level)

    tmp = path.with_name(path.name + ".partial")
    with h5py.File(tmp, "w") as f, ThreadPoolExecutor(workers) as pool:
        f.attrs.update({
            "format": FORMAT, "format_version": FORMAT_VERSION,
            "source_file": xf.path.name, "source_size": xf.path.stat().st_size,
            "kind": xf.kind, "created": datetime.now(timezone.utc).isoformat(),
            "writer": "xradia_reader " + _version(),
        })
        d = f.create_dataset("data", shape=(n, h, w), dtype=dt, chunks=(1, h, w),
                             shuffle=True, compression="gzip", compression_opts=level)
        d.attrs.update({
            "element_size_um": np.array([px if xf.is_volume else 1.0, px, px]),
            "unit": "um", "axes": "ZYX",
            "is_volume": xf.is_volume,
            "drop_bits": drop_bits, "quantization_step": 1 << drop_bits,
            "noise_sigma_estimate": sigma if sigma is not None else np.nan,
        })
        _write_imagej_attrs(d, xf.path.stem)
        # Read images in the main thread (olefile is not thread safe),
        # compress in the pool, write chunks in order.
        batch = 2 * workers
        for start in range(0, n, batch):
            imgs = [xf.image(idx[j]) for j in range(start, min(n, start + batch))]
            for img in imgs:
                src_hash.update(img.tobytes())
            for k, (q, err, chunk) in enumerate(pool.map(compress, imgs)):
                out_hash.update(q.tobytes())
                max_err = max(max_err, err)
                d.id.write_direct_chunk((start + k, 0, 0), chunk)
            if progress:
                progress(min(n, start + batch), n)
        d.attrs["source_pixel_sha256"] = src_hash.hexdigest()
        d.attrs["pixel_sha256"] = out_hash.hexdigest()
        d.attrs["max_abs_change"] = max_err

        ref = xf.reference
        if ref is not None:
            f.create_dataset("reference", data=ref, chunks=ref.shape,
                             shuffle=True, compression="gzip", compression_opts=level)

        summary = json.loads(json.dumps(xf.summary, default=_json_default))
        _write_key_values(f, summary)
        _put_json(f, "metadata", {"summary": summary, "tree": xf.metadata})
        _put_json(f, "raw_streams", {
            "/".join(p): xf.raw("/".join(p))
            for p in xf._ole.listdir(streams=True, storages=False) if not _is_pixel_stream(p)})
        t_write = time.time() - t0
        if source_hash:
            f.attrs["source_sha256"] = _sha256_file(xf.path)
    os.replace(tmp, path)

    report = {
        "source": str(xf.path), "archive": str(path),
        "source_bytes": xf.path.stat().st_size, "archive_bytes": path.stat().st_size,
        "drop_bits": drop_bits, "noise_sigma": sigma, "max_abs_change": max_err,
        "write_s": round(t_write, 1),
    }
    report["ratio"] = round(report["source_bytes"] / report["archive_bytes"], 3)
    if verify:
        t1 = time.time()
        report["verified"] = verify_hdf5(path)["ok"]
        report["verify_s"] = round(time.time() - t1, 1)
    return report


def verify_hdf5(path) -> dict:
    """Decode every image of an archive and check it against the stored
    SHA-256 (and, for lossless archives, the source pixel hash)."""
    with h5py.File(path, "r") as f:
        d = f["data"]
        h = hashlib.sha256()
        for i in range(d.shape[0]):
            h.update(np.ascontiguousarray(d[i]).tobytes())
        got = h.hexdigest()
        res = {"pixel_sha256_ok": got == d.attrs["pixel_sha256"]}
        if int(d.attrs.get("drop_bits", 0)) == 0:
            res["matches_source"] = got == d.attrs["source_pixel_sha256"]
    res["ok"] = all(res.values())
    return res


def _version():
    from . import __version__
    return __version__


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
class ArchiveFile(ImageStack):
    """Read an archive written by :func:`write_hdf5`; same API as XradiaFile
    (indexing, ``read``, ``summary``, ``metadata``, ``to_tiff``...)."""

    def __init__(self, path):
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")
        if self._h5.attrs.get("format") != FORMAT:
            raise ValueError(f"{path} is not an {FORMAT} file")
        self._data = self._h5["data"]

    def close(self):
        self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"<ArchiveFile {self.path.name!r} kind={self.kind} images={self.n_images} "
                f"shape={self.image_shape} dtype={self.dtype} drop_bits={self.drop_bits}>")

    @property
    def kind(self):
        return str(self._h5.attrs.get("kind", ""))

    @property
    def n_images(self):
        return self._data.shape[0]

    @property
    def image_shape(self):
        return tuple(self._data.shape[1:])

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def drop_bits(self) -> int:
        return int(self._data.attrs.get("drop_bits", 0))

    @property
    def pixel_size_um(self):
        return float(self._data.attrs["element_size_um"][2])

    @property
    def is_volume(self):
        return bool(self._data.attrs.get("is_volume", False))

    def image(self, i: int) -> np.ndarray:
        return self._data[i]

    def read(self, sel=None, dtype=None, out=None, scaled=False):
        if not scaled and out is None and (sel is None or isinstance(sel, slice)):
            a = self._data[sel if sel is not None else slice(None)]
            return a.astype(dtype) if dtype else a
        return super().read(sel, dtype, out, scaled)

    @property
    def reference(self):
        return self._h5["reference"][()] if "reference" in self._h5 else None

    @property
    def format_version(self) -> int:
        return int(self._h5.attrs.get("format_version", 1))

    @cached_property
    def _meta_doc(self) -> dict:
        if self.format_version >= 2:
            return _get_json(self._h5["metadata"])
        return {"summary": json.loads(self._h5["summary"].asstr()[()]),
                "tree": _read_tree(self._h5["metadata"])}

    @property
    def summary(self) -> dict:
        return self._meta_doc["summary"]

    @property
    def key_values(self) -> dict:
        """The important values (attributes of /summary in version 2)."""
        if self.format_version >= 2:
            return {k: (v.item() if isinstance(v, np.generic) else v)
                    for k, v in self._h5["summary"].attrs.items()}
        return key_values(self.summary)

    @property
    def metadata(self) -> dict:
        return self._meta_doc["tree"]

    def value(self, path, default=None):
        return _get_path(self.metadata, path, default)

    @cached_property
    def _raw(self) -> dict:
        if self.format_version >= 2:
            return _get_json(self._h5["raw_streams"])
        out = {}
        self._h5["raw_streams"].visititems(
            lambda name, o: out.__setitem__(name, _get(o)) if isinstance(o, h5py.Dataset) else None)
        return out

    def raw(self, path) -> bytes:
        return self._raw[path]

    def _global_minmax(self):
        return self.value("GlobalMinMax/GlobalMin"), self.value("GlobalMinMax/GlobalMax")


def upgrade_archive(src, dst=None, verify: bool = True) -> dict:
    """Bring an older archive to the current layout.

    Version 2 only lacks the ImageJ attributes on /data; they are added in
    place (or on a copy if ``dst`` is given).  Version 1 is rewritten; pixel
    chunks are copied as they are (no recompression).  ``dst=None`` replaces
    ``src``."""
    src = Path(src)
    dst = Path(dst) if dst else src
    with ArchiveFile(src) as a:
        version = a.format_version
    if version >= FORMAT_VERSION and dst == src:
        return {"archive": str(src), "upgraded": False}
    stamp = {"format_version": FORMAT_VERSION,
             "upgraded": datetime.now(timezone.utc).isoformat(),
             "writer": "xradia_reader " + _version()}
    if version == 2:
        if dst != src:
            shutil.copyfile(src, dst)
        with h5py.File(dst, "r+") as f:
            _write_imagej_attrs(f["data"], Path(str(f.attrs.get("source_file", dst.name))).stem)
            f.attrs.update(stamp)
        return {"archive": str(dst), "upgraded": True}

    tmp = dst.with_name(dst.name + ".partial")
    with ArchiveFile(src) as a:
        old = a._h5
        with h5py.File(tmp, "w") as f:
            f.attrs.update(dict(old.attrs))
            f.attrs.update(stamp)
            for name in ("data", "reference"):
                if name in old:
                    old.copy(old[name], f, name)  # H5Ocopy: raw chunks
            _write_imagej_attrs(f["data"], Path(str(old.attrs.get("source_file", dst.name))).stem)
            _write_key_values(f, a.summary)
            _put_json(f, "metadata", {"summary": a.summary, "tree": a.metadata})
            _put_json(f, "raw_streams", a._raw)
    report = {"archive": str(dst), "upgraded": True}
    if verify:
        report["verified"] = verify_hdf5(tmp)["ok"]
        if not report["verified"]:
            tmp.unlink()
            return report
    os.replace(tmp, dst)
    return report


def open_any(path) -> ImageStack:
    """Open an Xradia OLE file or an HDF5 archive."""
    return ArchiveFile(path) if h5py.is_hdf5(path) else XradiaFile(path)
