"""Reader for Zeiss Xradia .xrm / .txrm / .txm files (OLE2 compound documents).

Layout (as written by Scout-and-Scan / Reconstructor, e.g. Versa 620):

* ``ImageData{k}/Image{i}``  - image ``i`` (1-based), ``k = ceil(i / 100)``;
  raw pixels, row-major ``(ImageHeight, ImageWidth)``.
* ``ReferenceData/Image``    - flat-field reference (projections only).
* ``ImageInfo/*``            - acquisition info, scalars and per-image arrays.
* ``PositionInfo/*``         - motor table (per image x per axis).
* ``ReconSettings``, ``ReconInputTomoParams``, ``AutoRecon`` - recon (.txm).
* ``DetAssemblyInfo``, ``SampleInfo``, ``Alignment``, ``ConfigureBackup``...

Every stream is raw bytes without type information; see :mod:`.decode`.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from datetime import datetime
from functools import cached_property
from pathlib import Path

import numpy as np
import olefile

from .decode import decode_stream, guess_group_size, parse_xradia_date

#: ImageInfo/DataType codes (same as dxchange / txrm2tiff).
DATA_TYPES = {5: np.dtype("<u2"), 10: np.dtype("<f4")}
_BY_ITEMSIZE = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<f4")}

# Streams that hold pixel data, never decoded as metadata.
_MAX_META_STREAM = 4 * 1024 * 1024


def _is_pixel_stream(parts: list[str]) -> bool:
    top = parts[0].lower()
    if top.startswith("imagedata") and len(parts) == 2:
        return True
    return top == "referencedata" and parts[-1].lower() == "image"


class ImageStack:
    """Shared behaviour of image containers (Xradia OLE files and HDF5
    archives).  Subclasses provide ``image(i)``, ``n_images``,
    ``image_shape``, ``dtype``, ``summary``, ``metadata``, ``pixel_size_um``,
    ``is_volume`` and ``_global_minmax()``."""

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_images, *self.image_shape)

    def __len__(self):
        return self.n_images

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.image(int(key))
        if isinstance(key, tuple):
            first, rest = key[0], key[1:]
            return self[first][(slice(None), *rest) if not isinstance(first, (int, np.integer)) else rest]
        return self.read(key)

    def read(self, sel=None, dtype=None, out=None, scaled=False) -> np.ndarray:
        """Read several images into a 3-D array (n, height, width).

        ``sel`` is a slice, a sequence of indices or None (all images).
        ``dtype`` converts on the fly; ``out`` may be a preallocated array or
        ``np.memmap`` (useful for volumes larger than RAM).
        ``scaled=True`` returns float32 values mapped through
        :meth:`scale` (integer reconstructions only).
        """
        idx = range(self.n_images) if sel is None else (
            range(*sel.indices(self.n_images)) if isinstance(sel, slice) else list(sel))
        h, w = self.image_shape
        if out is None:
            out = np.empty((len(idx), h, w), np.float32 if scaled else (dtype or self.dtype))
        for k, i in enumerate(idx):
            im = self.image(i)
            out[k] = self.scale(im) if scaled else im
        return out

    def scale(self, im: np.ndarray) -> np.ndarray:
        """Map integer grey values of a reconstruction to the float range
        ``GlobalMinMax/GlobalMin .. GlobalMax`` (linear over the full dtype
        range).  This mapping is *inferred* (air lands close to 0 in the
        sample data); Zeiss does not document it.  Pixels outside the
        reconstructed field of view are 0 and map to GlobalMin.
        """
        lo, hi = self._global_minmax()
        if lo is None or hi is None or im.dtype.kind not in "ui":
            return im.astype(np.float32)
        top = np.iinfo(im.dtype).max
        return (lo + im.astype(np.float32) * ((hi - lo) / top)).astype(np.float32)

    def __iter__(self):
        for i in range(self.n_images):
            yield self.image(i)

    def to_tiff(self, path, sel=None, per_slice: bool = False, info: bool = True):
        """Write images as ImageJ/Fiji-calibrated TIFF; pixel values untouched.

        The data keep their stored dtype (uint16 for Versa reconstructions),
        exactly as dxchange returns them; only spatial calibration is added:
        XY resolution tags + ``unit=um`` and, for volumes, the Z ``spacing``
        (= voxel size), so Fiji shows Image > Properties in micrometres.
        With ``info`` the summary is stored in ImageJ's Info property
        (Image > Show Info).

        ``per_slice=True`` writes ``<stem>_00000.tif`` ... (one file per
        image, each calibrated) instead of a single multi-page stack.
        Slices are streamed, so the volume is never held in memory.
        Returns the list of written paths.
        """
        import tifffile

        idx = range(self.n_images) if sel is None else (
            range(*sel.indices(self.n_images)) if isinstance(sel, slice) else list(sel))
        h, w = self.image_shape
        px = self.pixel_size_um
        is_volume = self.is_volume
        md = {"axes": "ZYX"}
        kw = {}
        if px:
            md["unit"] = "um"
            kw["resolution"] = (1.0 / px, 1.0 / px)
            if is_volume:
                md["spacing"] = px
        if info:
            md["Info"] = _info_text(self.summary)

        path = Path(path)
        if not per_slice:
            tifffile.imwrite(path, data=(self.image(i) for i in idx),
                             shape=(len(idx), h, w), dtype=self.dtype,
                             imagej=True, metadata=md, **kw)
            return [path]
        out = []
        digits = max(5, len(str(self.n_images)))
        for i in idx:
            p = path.with_name(f"{path.stem}_{i:0{digits}d}{path.suffix or '.tif'}")
            m = dict(md, axes="YX")
            tifffile.imwrite(p, self.image(i), imagej=True, metadata=m, **kw)
            out.append(p)
        return out

    def to_json(self, path=None, full: bool = True, indent: int = 1):
        """Serialise summary (+ full metadata tree if ``full``) to JSON."""
        doc = {"summary": self.summary}
        if full:
            doc["metadata"] = self.metadata
        s = json.dumps(doc, indent=indent, default=_json_default)
        if path is not None:
            Path(path).write_text(s)
        return s


class XradiaFile(ImageStack):
    """An open Xradia file.

    >>> with XradiaFile("scan.txrm") as f:
    ...     f.shape, f.dtype
    ...     proj = f[0]              # first image, 2-D array
    ...     stack = f[100:200:10]    # 3-D array, only those images are read
    ...     f.summary["source"]["voltage_kV"]
    ...     f.metadata["ImageInfo"]["Angles"]
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._ole = olefile.OleFileIO(str(self.path))
        self._meta = None
        self._summary = None
        self._ctx = None

    # ------------------------------------------------------------------ basics
    def close(self):
        self._ole.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return (f"<XradiaFile {self.path.name!r} kind={self.kind} "
                f"images={self.n_images} shape={self.image_shape} dtype={self.dtype}>")

    @property
    def kind(self) -> str:
        """'xrm' (single image / mosaic), 'txrm' (projections) or 'txm'
        (reconstructed volume), from the file extension."""
        return self.path.suffix.lower().lstrip(".")

    def streams(self) -> list[str]:
        """All stream paths, e.g. 'ImageInfo/Angles'."""
        return ["/".join(p) for p in self._ole.listdir(streams=True, storages=False)]

    def has(self, path: str) -> bool:
        return self._ole.exists(path)

    def raw(self, path: str) -> bytes:
        """Raw bytes of a stream."""
        with self._ole.openstream(path) as s:
            return s.read()

    def value(self, path: str, default=None):
        """Decoded value of one stream (``default`` if missing)."""
        if not self._ole.exists(path):
            return default
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        return decode_stream(path, self.raw(path), self._context(parent))

    def _context(self, storage: str) -> dict:
        if self._ctx is None:
            n_ax = None
            if self._ole.exists("PositionInfo/TotalAxis"):
                n_ax = int(np.frombuffer(self.raw("PositionInfo/TotalAxis")[:4], "<i4")[0])
            self._ctx = {"n_axes": n_ax, "groups": {}}
        groups = self._ctx["groups"]
        if storage not in groups:
            sizes = {}
            for p in self._ole.listdir(streams=True, storages=False):
                if "/".join(p[:-1]) == storage:
                    sizes[p[-1]] = self._ole.get_size("/".join(p))
            groups[storage] = guess_group_size(sizes)
        return {"n_axes": self._ctx["n_axes"], "group": groups[storage]}

    # ------------------------------------------------------------- image info
    def _int(self, path, default=None):
        v = self.value(path)
        return default if v is None else int(v)

    @cached_property
    def n_images(self) -> int:
        n = self._int("ImageInfo/NoOfImages")
        if n is None:
            n = sum(1 for p in self._ole.listdir(streams=True, storages=False)
                    if p[0].lower().startswith("imagedata") and len(p) == 2)
        return n

    @cached_property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) of one image."""
        return (self._int("ImageInfo/ImageHeight"), self._int("ImageInfo/ImageWidth"))

    @cached_property
    def _datatype_code(self):
        return self._int("ImageInfo/DataType")

    @property
    def pixel_size_um(self):
        return self.value("ImageInfo/PixelSize")

    @property
    def is_volume(self) -> bool:
        return self.kind == "txm"

    def _global_minmax(self):
        return self.value("GlobalMinMax/GlobalMin"), self.value("GlobalMinMax/GlobalMax")

    def _dtype_for(self, code, nbytes) -> np.dtype:
        h, w = self.image_shape
        by_size = _BY_ITEMSIZE.get(nbytes // (h * w)) if h and w else None
        dt = DATA_TYPES.get(code)
        if dt is None:
            if by_size is None:
                raise ValueError(f"unknown DataType {code} and cannot infer from size")
            return by_size
        if by_size is not None and by_size.itemsize != dt.itemsize:
            warnings.warn(f"DataType {code} ({dt}) disagrees with stream size; using {by_size}")
            return by_size
        return dt

    @cached_property
    def dtype(self) -> np.dtype:
        first = self._image_stream(0)
        nbytes = self._ole.get_size(first) if first else 0
        return self._dtype_for(self._datatype_code, nbytes)

    @staticmethod
    def _image_path(i: int) -> str:
        return f"ImageData{math.ceil((i + 1) / 100)}/Image{i + 1}"

    def _image_stream(self, i: int) -> str | None:
        p = self._image_path(i)
        return p if self._ole.exists(p) else None

    # --------------------------------------------------------------- pixels
    def image(self, i: int) -> np.ndarray:
        """Image ``i`` (0-based) as a 2-D array of shape (height, width)."""
        n = self.n_images
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"image index {i} out of range [0, {n})")
        p = self._image_stream(i)
        if p is None:
            raise KeyError(f"stream {self._image_path(i)} missing")
        b = self.raw(p)
        h, w = self.image_shape
        dt = self.dtype if len(b) == h * w * self.dtype.itemsize else \
            self._dtype_for(self._datatype_code, len(b))
        return np.frombuffer(b, dt, count=h * w).reshape(h, w).copy()

    @property
    def reference(self) -> np.ndarray | None:
        """Flat-field reference image (ReferenceData/Image) or None."""
        if not self._ole.exists("ReferenceData/Image"):
            return None
        b = self.raw("ReferenceData/Image")
        code = self.value("ReferenceData/DataType", self._int("ImageInfo/DataType"))
        h, w = self.image_shape
        dt = self._dtype_for(code, len(b))
        return np.frombuffer(b, dt, count=h * w).reshape(h, w).copy()

    # -------------------------------------------------------------- metadata
    @property
    def metadata(self) -> dict:
        """Every non-pixel stream, decoded, as a nested dict mirroring the
        OLE tree.  Per-image arrays are trimmed to ``n_images``."""
        if self._meta is None:
            self._meta = self._read_all()
        return self._meta

    def _read_all(self) -> dict:
        root: dict = {}
        n_img = self.n_images
        for parts in self._ole.listdir(streams=True, storages=False):
            if _is_pixel_stream(parts):
                continue
            p = "/".join(parts)
            size = self._ole.get_size(p)
            if size > _MAX_META_STREAM:
                v = f"<{size} bytes, not decoded>"
            else:
                v = decode_stream(p, self.raw(p), self._context("/".join(parts[:-1])))
                v = _trim_per_image(parts, v, n_img)
            d = root
            for k in parts[:-1]:
                d = d.setdefault(k, {})
            d[parts[-1]] = v
        return root

    @property
    def summary(self) -> dict:
        """Curated, human-readable selection of the metadata with units and
        derived quantities (power, magnification, scan duration...)."""
        if self._summary is None:
            self._summary = _build_summary(self)
        return self._summary

def _trim_per_image(parts, v, n_img):
    top = parts[0].lower()
    if top not in ("imageinfo", "positioninfo", "alignment", "selection") or not n_img:
        return v
    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] > n_img:
        return v[:n_img]
    if isinstance(v, list) and len(v) > n_img and parts[-1].lower() == "date":
        return v[:n_img]
    return v


def _info_text(summary: dict) -> str:
    """Flatten the summary into 'section.key = value' lines for ImageJ."""
    lines = []

    def rec(d, prefix):
        for k, v in d.items():
            if isinstance(v, dict) and not set(v) <= {"min", "max", "mean", "n"}:
                rec(v, f"{prefix}{k}.")
            else:
                if isinstance(v, dict):
                    v = f"{v['min']} .. {v['max']} (mean {v['mean']}, n={v['n']})"
                elif isinstance(v, np.ndarray):
                    v = np.array2string(v, separator=", ", threshold=20)
                lines.append(f"{prefix}{k} = {v}")

    rec(summary, "")
    return "\n".join(lines)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, datetime):
        return o.isoformat()
    if isinstance(o, bytes):
        return {"hex": o.hex()}
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _get(meta, path, default=None):
    d = meta
    for k in path.split("/"):
        if not isinstance(d, dict):
            return default
        # OLE names are case-insensitive
        m = {kk.lower(): kk for kk in d}
        if k.lower() not in m:
            return default
        d = d[m[k.lower()]]
    return d


def _stat(a):
    """Collapse a per-image array: a scalar if constant, else min/max/mean."""
    if a is None:
        return None
    a = np.asarray(a)
    if a.size == 0:
        return None
    if a.dtype.kind == "f":
        finite = a[np.isfinite(a)]
        if finite.size == 0:
            return None
        a = finite
    conv = (lambda x: float(str(np.float32(x)))) if a.dtype == np.float32 else (lambda x: x.item())
    if np.all(a == a.flat[0]):
        return conv(a.flat[0])
    return {"min": conv(a.min()), "max": conv(a.max()),
            "mean": conv(a.mean(dtype=np.float64).astype(a.dtype)), "n": int(a.size)}


def _nonzero(a):
    a = None if a is None else np.asarray(a)
    return a if a is not None and a.size and np.any(a != 0) else None


def _drop_none(d):
    if isinstance(d, dict):
        out = {k: _drop_none(v) for k, v in d.items()}
        return {k: v for k, v in out.items() if v is not None and not (isinstance(v, dict) and not v)}
    return d


def _build_summary(f: XradiaFile) -> dict:
    m = f.metadata
    g = lambda p, d=None: _get(m, p, d)  # noqa: E731
    n = f.n_images
    h, w = f.image_shape

    # --- dates
    dates = [parse_xradia_date(s) for s in (g("ImageInfo/Date") or []) if s]
    dates = [d for d in dates if d]
    t0, t1 = (min(dates), max(dates)) if dates else (None, None)

    # --- geometry
    s2ra = _stat(g("ImageInfo/StoRADistance"))
    d2ra = _stat(g("ImageInfo/DtoRADistance"))
    geo_mag = None
    if isinstance(s2ra, (int, float)) and isinstance(d2ra, (int, float)) and s2ra:
        geo_mag = (abs(s2ra) + abs(d2ra)) / abs(s2ra)
    opt_mag = g("ImageInfo/OpticalMagnification")
    cam_px = g("ImageInfo/CamPixelSize")
    binning = g("ImageInfo/CameraBinning")
    px_derived = None
    if geo_mag and opt_mag and cam_px and binning:
        px_derived = cam_px * binning / (opt_mag * geo_mag)

    kv, ua = g("ImageInfo/Voltage"), g("ImageInfo/Current")

    # --- motors
    motors = None
    names = g("PositionInfo/AxisNames")
    pos = g("PositionInfo/MotorPositions")
    if names and isinstance(pos, np.ndarray) and pos.ndim == 2:
        units = g("PositionInfo/AxisUnits") or [""] * len(names)
        motors = {}
        for j, name in enumerate(names):
            if not name:
                continue
            motors[name] = {"unit": units[j] if j < len(units) else "",
                            "value": _stat(pos[:, j])}

    angles = g("ImageInfo/Angles")
    is_recon = f.kind == "txm" or g("ReconSettings") is not None

    s = {
        "file": {
            "path": str(f.path), "kind": f.kind,
            "size_bytes": f.path.stat().st_size,
            "software_version": g("exeVersion"),
            "file_version": g("Version"),
            "data_bundle_version": g("DataBundleVer"),
        },
        "image": {
            "n_images": n, "height": h, "width": w,
            "dtype": str(f.dtype),
            "pixel_size_um": g("ImageInfo/PixelSize"),
            "voxel_size_um": g("ImageInfo/PixelSize") if is_recon else None,
            "mosaic_rows": g("ImageInfo/MosiacRows") or None,
            "mosaic_columns": g("ImageInfo/MosiacColumns") or None,
            "global_min": g("GlobalMinMax/GlobalMin"),
            "global_max": g("GlobalMinMax/GlobalMax"),
            "transmission_scale_factor": g("ImageInfo/TransmissionScaleFactor"),
            "absorption_scale_factor": g("ImageInfo/AbsorptionScaleFactor"),
            "absorption_scale_offset": g("ImageInfo/AbsorptionScaleOffset"),
        },
        "source": {
            "voltage_kV": kv,
            "current_uA": ua,
            "power_W": round(kv * ua / 1000, 4) if kv and ua else None,
            "filter": g("ImageInfo/SourceFilterName"),
            "filter_id": g("ImageInfo/SourceFilterID"),
            "per_image_voltage_kV": _stat(_nonzero(g("ImageInfo/XrayVoltage"))),
            "per_image_current_uA": _stat(_nonzero(g("ImageInfo/XrayCurrent"))),
            "spot_offset_um": [g("ImageInfo/NetSpotOffsetBLXum"), g("ImageInfo/NetSpotOffsetBLYum")]
            if g("ImageInfo/NetSpotOffsetBLXum") is not None else None,
        },
        "geometry": {
            "source_to_rotation_axis_um": s2ra,
            "detector_to_rotation_axis_um": d2ra,
            "geometric_magnification": round(geo_mag, 6) if geo_mag else None,
            "optical_magnification": opt_mag,
            "total_magnification": round(geo_mag * opt_mag, 6) if geo_mag and opt_mag else None,
            "pixel_size_derived_um": round(px_derived, 5) if px_derived else None,
            "cone_angle_deg": _stat(g("ImageInfo/ConeAngle")),
            "fan_angle_deg": _stat(g("ImageInfo/FanAngle")),
            "src_to_ra_config_um": g("ImageInfo/SrcToRA"),
            "det_to_ra_config_um": g("ImageInfo/DetToRA"),
        },
        "detector": {
            "camera": g("ImageInfo/CameraName"),
            "serial": g("ImageInfo/CameraSerialNumberString"),
            "objective": g("ImageInfo/ObjectiveName"),
            "objective_id": g("ImageInfo/ObjectiveID"),
            "camera_pixel_size_um": cam_px,
            "sensor_width": g("ImageInfo/CamFullWidth"),
            "sensor_height": g("ImageInfo/CamFullHeight"),
            "binning": binning,
            "temperature_setpoint_C": g("ImageInfo/Temperature"),
            "readout_frequency_Hz": g("ImageInfo/ReadoutFreq"),
            "sensitivity": g("ImageInfo/CameraSensitivity"),
            "flat_panel": g("ImageInfo/IsFlatPanel"),
            "lens_distortion_enabled": g("DetAssemblyInfo/LensInfo/DistortEnabled"),
            "lens_distortion_params": g("DetAssemblyInfo/LensInfo/DistortParam"),
        },
        "acquisition": {
            # In a reconstructed volume ImageInfo/Date holds the time each
            # slice was written, not the scan time.
            "start" if not is_recon else "slices_written_start": t0,
            "end" if not is_recon else "slices_written_end": t1,
            "duration_s": (t1 - t0).total_seconds() if dates else None,
            "exposure_s": _stat(g("ImageInfo/ExpTimes")),
            "frames_per_image": _stat(g("ImageInfo/FramesPerImage")),
            "images_averaged": g("ImageInfo/NoOfImagesAveraged"),
            "angles_deg": None if is_recon else _stat(angles),
            "angle_step_deg": None if is_recon or angles is None or len(angles) < 2
            else float(np.median(np.diff(angles))),
            "reference_interval": g("ImageInfo/RefInterval"),
            "reference_file": g("ImageInfo/ReferenceFile") or None,
            "reference_present": f.has("ReferenceData/Image") or None,
            "motors": motors,
        },
        "sample": g("SampleInfo"),
        "alignment": {
            "x_shifts_px": _stat(g("Alignment/X-Shifts")),
            "y_shifts_px": _stat(g("Alignment/Y-Shifts")),
        } if g("Alignment") else None,
    }

    if is_recon:
        rs = lambda k: g("ReconSettings/" + k)  # noqa: E731
        rp = lambda k: g("ReconInputTomoParams/" + k)  # noqa: E731
        av = rp("AngleVector")
        ys = g("ImageInfo/YPosition")
        s["reconstruction"] = {
            "input_file": rs("InputFileName"),
            "output_file": rs("OutputFileLocation"),
            "n_projections": rp("NumOfProjects"),
            "angle_span_deg": rp("AngleSpan"),
            "projection_angles_deg": _stat(av),
            "projection_binning": rp("Binning"),
            "recon_binning": rs("ReconBinning"),
            "projection_crop": [rp("CroppedHeight"), rp("CroppedWidth")],
            "center_shift_px": rs("CenterShift"),
            "auto_center_shift": g("AutoRecon/AutoCenterShift"),
            "rotation_angle_deg": rs("RotationAngle"),
            "beam_hardening": rs("BeamHardening"),
            "beam_hardening_file": rs("BeamHardeningFileName"),
            "filter_choice": rs("ReconFilterChoice"),
            "filter_smooth_factor": rs("ReconFilterSmoothFactor"),
            "ring_removal": rs("RemoveRingON"),
            "defect_correction": rs("DefectCorrection"),
            "global_min": rs("GlobalMin"), "global_max": rs("GlobalMax"),
            "source_filter": rs("SourceFilter"),
            "source_voltage_kV": rs("SourceVoltage"),
            "mean_sample_xyz_um": [rp("MeanSampleX"), rp("MeanSampleY"), rp("MeanSampleZ")],
            "auto_stitch": _get(m, "ReconSettings/StitchParams/AutoStitchSettings/UseAutoStitch"),
            # For volumes the per-slice Y position is the slice coordinate.
            "slice_y_um": _stat(ys),
            "slice_spacing_um": float(np.median(np.diff(ys))) if ys is not None and len(ys) > 1 else None,
        }
    return _drop_none(s)


#: The few values worth seeing at a glance: (label, summary path).
KEY_VALUES = [
    ("kind", "file/kind"),
    ("software_version", "file/software_version"),
    ("n_images", "image/n_images"),
    ("height", "image/height"),
    ("width", "image/width"),
    ("dtype", "image/dtype"),
    ("pixel_size_um", "image/pixel_size_um"),
    ("voltage_kV", "source/voltage_kV"),
    ("current_uA", "source/current_uA"),
    ("power_W", "source/power_W"),
    ("source_filter", "source/filter"),
    ("objective", "detector/objective"),
    ("binning", "detector/binning"),
    ("source_to_rotation_axis_um", "geometry/source_to_rotation_axis_um"),
    ("detector_to_rotation_axis_um", "geometry/detector_to_rotation_axis_um"),
    ("total_magnification", "geometry/total_magnification"),
    ("exposure_s", "acquisition/exposure_s"),
    ("acquisition_start", "acquisition/start"),
    ("acquisition_end", "acquisition/end"),
    ("n_projections", "reconstruction/n_projections"),
    ("angle_span_deg", "reconstruction/angle_span_deg"),
    ("recon_binning", "reconstruction/recon_binning"),
    ("center_shift_px", "reconstruction/center_shift_px"),
    ("beam_hardening", "reconstruction/beam_hardening"),
    ("recon_input_file", "reconstruction/input_file"),
]


def key_values(summary: dict) -> dict:
    """Flat dict of the most important summary values (see KEY_VALUES).
    Missing entries are skipped; per-image ranges become 'min .. max'."""
    out = {}
    for label, path in KEY_VALUES:
        v = _get(summary, path)
        if v is None:
            continue
        if isinstance(v, dict) and {"min", "max"} <= set(v):
            v = f"{v['min']} .. {v['max']}"
        elif isinstance(v, datetime):
            v = v.isoformat(sep=" ")
        out[label] = v
    return out


# ---------------------------------------------------------------------------
# functional API
# ---------------------------------------------------------------------------
def read(path, sel=None):
    """Read images and metadata: returns ``(array, summary, metadata)``.

    For a single-image .xrm the array is 2-D.
    """
    with XradiaFile(path) as f:
        data = f.read(sel)
        if f.n_images == 1 and sel is None:
            data = data[0]
        return data, f.summary, f.metadata


def read_metadata(path) -> dict:
    """Full decoded metadata tree (no pixel data is read)."""
    with XradiaFile(path) as f:
        return f.metadata
