import numpy as np

from xradia_reader.decode import decode_heuristic, decode_stream, guess_group_size


def test_scalars():
    assert decode_heuristic(np.int32(1014).tobytes()) == 1014
    assert decode_heuristic(np.int32(-60).tobytes()) == -60
    assert decode_heuristic(np.float32(0.39157).tobytes()) == 0.39157
    assert decode_heuristic(b"\x01") is True


def test_strings():
    assert decode_heuristic(b"Andor0".ljust(80, b"\0")) == "Andor0"
    # empty string followed by garbage, recognised by name
    junk = b"\0" + bytes(range(1, 256)) + b"\x7f" * 4
    assert decode_stream("ImageInfo/ReferenceFile", junk) == ""
    # float bytes that happen to be printable are not text
    b = np.array([0.0487, -0.0487], "<f4").tobytes()
    assert isinstance(decode_heuristic(b), np.ndarray)


def test_arrays():
    a = np.arange(5, dtype="<i4")
    assert decode_heuristic(a.tobytes()).dtype == np.int32
    f = np.linspace(-180, 180, 7, dtype="<f4")
    np.testing.assert_array_equal(decode_heuristic(f.tobytes()), f)


def test_table_columns():
    sizes = {"A": 24, "B": 24, "C": 24, "Name": 150}
    g = guess_group_size(sizes)
    assert g == 6
    names = b"".join(n.ljust(25, b"\0") for n in [b"4X", b"20X", b"0.4X", b"", b"", b""])
    assert decode_stream("X/ObjectiveName", names, {"group": g}) == ["4X", "20X", "0.4X", "", "", ""]
    flags = bytes([0, 1, 1, 0, 0, 0])
    assert decode_stream("X/Flags", flags, {"group": g}).tolist() == [False, True, True, False, False, False]
