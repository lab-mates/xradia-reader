"""Tests against the files in sample_data/ (skipped when absent)."""
from pathlib import Path

import numpy as np
import pytest

from xradia_reader import XradiaFile

SAMPLE = Path(__file__).parent.parent / "sample_data" / "REF_1000C.txm"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="sample data not present")


@pytest.fixture(scope="module")
def f():
    with XradiaFile(SAMPLE) as x:
        yield x


def test_shape(f):
    assert f.shape == (1014, 1024, 1002)
    assert f.dtype == np.uint16
    assert f[0].shape == (1024, 1002)
    assert f[10:13].shape == (3, 1024, 1002)
    assert f[5, 100:110, 0:4].shape == (10, 4)


def test_summary(f):
    s = f.summary
    assert s["source"]["voltage_kV"] == 60.0
    assert s["detector"]["objective"] == "0.4X"
    assert s["reconstruction"]["n_projections"] == 2401
    # pixel size from geometry agrees with the stored one
    assert abs(s["geometry"]["pixel_size_derived_um"] - s["image"]["pixel_size_um"]) < 1e-3
    assert list(s["acquisition"]["motors"])[:4] == ["Sample X", "Sample Y", "Sample Z", "Sample Theta"]


def test_metadata_tree(f):
    m = f.metadata
    assert m["ImageInfo"]["Angles"].shape == (1014,)
    assert m["PositionInfo"]["MotorPositions"].shape == (1014, 9)
    assert len(m["ImageInfo"]["Date"]) == 1014
    assert m["ConfigureBackup"]["ConfigMotion"]["AxisName"][3] == "Sample Theta"
    assert m["ReconInputTomoParams"]["AngleVector"].shape == (2401,)


def test_json(f):
    import json
    d = json.loads(f.to_json())
    assert d["summary"]["image"]["n_images"] == 1014


def test_tiff(f, tmp_path):
    tifffile = pytest.importorskip("tifffile")
    p = f.to_tiff(tmp_path / "s.tif", sel=slice(500, 503))[0]
    with tifffile.TiffFile(p) as t:
        md = t.imagej_metadata
        assert md["unit"] == "um" and abs(md["spacing"] - 20.111942) < 1e-5
        num, den = t.pages[0].tags["XResolution"].value
        assert abs(den / num - 20.111942) < 1e-4
        np.testing.assert_array_equal(t.asarray(), f[500:503])
