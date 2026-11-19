from pathlib import Path

import h5py
import numpy as np
import pytest

from xradia_reader import ArchiveFile, XradiaFile, open_any, verify_hdf5, write_hdf5
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
