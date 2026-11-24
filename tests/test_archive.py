from pathlib import Path

import h5py
import numpy as np
import pytest

from xradia_reader import ArchiveFile, XradiaFile, open_any, upgrade_archive, verify_hdf5, write_hdf5
from xradia_reader.archive import auto_drop_bits, quantize

SAMPLE = Path(__file__).parent.parent / "sample_data" / "REF_1000C.txm"
needs_sample = pytest.mark.skipif(not SAMPLE.exists(), reason="sample data not present")


def test_quantize_bounds():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 65536, (64, 64)).astype(np.uint16)
    a[:5] = 0
    a[5, :3] = 65535
    for k in (1, 5, 8):
        q = quantize(a, k)
        assert q.dtype == np.uint16
        assert np.abs(q.astype(int) - a).max() <= 1 << (k - 1)
        assert (q[:5] == 0).all()
        nz = q[(a > 0) & (q != 65535)]
        assert (nz % (1 << k) == 0).all()
    assert quantize(a, 0) is a


def test_auto_drop_bits():
    assert auto_drop_bits(None) == 0
    assert auto_drop_bits(10) == 0
    assert auto_drop_bits(907) == 5
    assert auto_drop_bits(1850) == 6
    assert auto_drop_bits(1e6) == 8


@pytest.fixture(scope="module")
def archives(tmp_path_factory):
    d = tmp_path_factory.mktemp("arch")
    with XradiaFile(SAMPLE) as xf:
        r0 = write_hdf5(xf, d / "l.h5", sel=slice(500, 506), source_hash=False)
        r5 = write_hdf5(xf, d / "q.h5", drop_bits=5, sel=slice(500, 506), source_hash=False)
    return d, r0, r5


@needs_sample
def test_lossless_roundtrip(archives):
    d, r0, _ = archives
    assert r0["verified"] and r0["max_abs_change"] == 0
    with XradiaFile(SAMPLE) as xf, open_any(d / "l.h5") as a:
        assert isinstance(a, ArchiveFile)
        np.testing.assert_array_equal(a[:], xf[500:506])
        np.testing.assert_array_equal(a[2, 10:20], xf[502, 10:20])
        assert a.summary["image"]["pixel_size_um"] == xf.summary["image"]["pixel_size_um"]
        assert a.raw("ImageInfo/PixelSize") == xf.raw("ImageInfo/PixelSize")
        m = a.metadata
        assert m["ConfigureBackup"]["ConfigMotion"]["AxisName"][3] == "Sample Theta"
        assert m["ImageInfo"]["Date"] == xf.metadata["ImageInfo"]["Date"]
        assert m["ImageInfo"]["XrayPresent"] is True


@needs_sample
def test_near_lossless(archives):
    d, r0, r5 = archives
    assert r5["verified"] and 0 < r5["max_abs_change"] <= 16
    assert r5["archive_bytes"] < r0["archive_bytes"]
    with XradiaFile(SAMPLE) as xf, ArchiveFile(d / "q.h5") as a:
        assert a.drop_bits == 5
        assert np.abs(a[:].astype(int) - xf[500:506]).max() <= 16


@needs_sample
def test_plain_hdf5_readable(archives):
    d, _, _ = archives
    with h5py.File(d / "l.h5", "r") as f:  # no xradia_reader involved
        ds = f["data"]
        assert ds.compression == "gzip" and ds.shuffle
        assert ds.dtype == np.uint16
        assert abs(ds.attrs["element_size_um"][0] - 20.111942) < 1e-5


@needs_sample
def test_verify_detects_corruption(archives, tmp_path):
    d, _, _ = archives
    p = tmp_path / "c.h5"
    p.write_bytes((d / "l.h5").read_bytes())
    with h5py.File(p, "r+") as f:
        f["data"][0, 0, 0] = f["data"][0, 0, 0] ^ 1
    assert verify_hdf5(d / "l.h5")["ok"]
    assert not verify_hdf5(p)["ok"]


@needs_sample
def test_v2_layout_is_compact(archives):
    d, _, _ = archives
    with h5py.File(d / "l.h5", "r") as f:
        names = []
        f.visit(names.append)
        assert sorted(names) == ["data", "metadata", "raw_streams", "summary"]
        ds = f["data"]
        assert ds.attrs["pixelUnit"] == "um" and ds.attrs["numSlices"] == 6
        assert ds.attrs["pixelWidth"] == ds.attrs["pixelDepth"] == ds.attrs["element_size_um"][2]
        kv = dict(f["summary"].attrs)
        assert abs(kv["pixel_size_um"] - 20.111942) < 1e-5
        assert kv["voltage_kV"] == 60.0 and kv["objective"] == "0.4X"
        # metadata blob is plain JSON
        import json
        doc = json.loads(bytes(f["metadata"][()]))
        assert doc["summary"]["source"]["voltage_kV"] == 60.0


def _write_v1(path):
    """Minimal archive in the version-1 layout."""
    import hashlib
    import json
    data = np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)
    h = hashlib.sha256()
    for im in data:
        h.update(im.tobytes())
    s = h5py.string_dtype("utf-8")
    with h5py.File(path, "w") as f:
        f.attrs.update({"format": "xradia-archive", "format_version": 1, "kind": "txm"})
        ds = f.create_dataset("data", data=data, chunks=(1, 4, 5), compression="gzip", shuffle=True)
        ds.attrs.update({"element_size_um": [2.0, 2.0, 2.0], "is_volume": True, "drop_bits": 0,
                         "pixel_sha256": h.hexdigest(), "source_pixel_sha256": h.hexdigest()})
        f.create_dataset("summary", data=json.dumps(
            {"image": {"pixel_size_um": 2.0}, "source": {"voltage_kV": 80.0}}), dtype=s)
        m = f.create_group("metadata/ImageInfo")
        m.create_dataset("PixelSize", data=np.float32(2.0))
        m.create_dataset("Angles", data=np.linspace(0, 1, 3, dtype=np.float32))
        r = f.create_group("raw_streams/ImageInfo")
        r.create_dataset("PixelSize", data=np.frombuffer(np.float32(2.0).tobytes(), np.uint8)) \
            .attrs["py_type"] = "bytes"
    return data


def test_upgrade_v1(tmp_path):
    p = tmp_path / "old.h5"
    data = _write_v1(p)
    with ArchiveFile(p) as a:
        assert a.format_version == 1 and a.key_values["voltage_kV"] == 80.0
        assert a.raw("ImageInfo/PixelSize") == np.float32(2.0).tobytes()
    r = upgrade_archive(p)
    assert r["upgraded"] and r["verified"]
    with ArchiveFile(p) as a:
        assert a.format_version == 3
        assert a._data.attrs["pixelUnit"] == "um" and a._data.attrs["pixelDepth"] == 2.0
        np.testing.assert_array_equal(a[:], data)
        assert a._data.compression == "gzip"
        assert a.key_values == {"pixel_size_um": 2.0, "voltage_kV": 80.0}
        assert a.value("ImageInfo/PixelSize") == 2.0
        np.testing.assert_array_equal(a.metadata["ImageInfo"]["Angles"],
                                      np.linspace(0, 1, 3, dtype=np.float32))
        assert a.metadata["ImageInfo"]["Angles"].dtype == np.float32
        assert a.raw("ImageInfo/PixelSize") == np.float32(2.0).tobytes()
    assert not upgrade_archive(p)["upgraded"]


@needs_sample
def test_upgrade_v2_in_place(archives, tmp_path):
    d, _, _ = archives
    p = tmp_path / "v2.h5"
    p.write_bytes((d / "l.h5").read_bytes())
    with h5py.File(p, "r+") as f:
        for k in ("pixelUnit", "pixelWidth", "numSlices"):
            del f["data"].attrs[k]
        f.attrs["format_version"] = 2
    assert upgrade_archive(p)["upgraded"]
    with h5py.File(p, "r") as f:
        assert f.attrs["format_version"] == 3
        assert f["data"].attrs["pixelUnit"] == "um" and f["data"].attrs["numSlices"] == 6
        assert f["data"].attrs["title"] == "REF_1000C"
    assert verify_hdf5(p)["ok"]
