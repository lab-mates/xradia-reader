"""Decoding of raw Xradia OLE streams into Python / NumPy values.

Xradia files are OLE2 compound documents in which every metadata item is a
separate stream holding raw little-endian bytes with *no type information*.
The type of each stream therefore has to be inferred.  We use:

1. a table of explicit decoders for streams whose layout is known
   (records of fixed-width strings, 2-D motor tables, ...);
2. a heuristic for everything else:

   * 1 byte                -> bool
   * ``n % 4 != 0``        -> string (NUL terminated, cp1252) or raw bytes
   * ``n % 4 == 0``        -> string if it looks like text, otherwise an
     array of 32-bit words that is int32 when *every* word, read as float32,
     is zero, denormal or NaN (the bit patterns small/negative integers
     produce) and float32 otherwise.

The heuristic cannot tell ``0`` (int) from ``0.0`` (float); such values are
returned as ints for scalars and float32 for arrays.
"""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np

STRING_ENCODING = "cp1252"

# Printable cp1252 text: tab/CR/LF, 0x20-0x7E and 0xA0-0xFF.
_TEXT_RE = re.compile(rb"^[\t\r\n\x20-\x7e\xa0-\xff]+$")


def _cstr(b: bytes) -> str:
    return b.split(b"\0", 1)[0].decode(STRING_ENCODING, errors="replace").strip()


_ASCII_TEXT = set(range(0x20, 0x7F)) | {9, 10, 13}


def _looks_like_text(b: bytes) -> bool:
    head = b.split(b"\0", 1)[0]
    if not head or not _TEXT_RE.match(head):
        return False
    # Float bit patterns are often "printable" (b'\xde\xffG='); require the
    # text to be mostly plain ASCII.
    if sum(c in _ASCII_TEXT for c in head) < 0.8 * len(head):
        return False
    if len(b) % 4:
        return True
    # A run of small integers (b'0\0\0\0...') also has a printable prefix, so
    # for word-aligned streams require a reasonably long prefix or a prefix
    # followed by NULs only.
    return len(head) >= 4 or not b[len(head):].strip(b"\0")


def _int_like(words: np.ndarray) -> bool:
    """True if 32-bit words are better read as int32 than float32."""
    u = words.view("<u4")
    exp = (u >> 23) & 0xFF
    # exponent 0 -> zero/denormal, 0xFF -> inf/NaN (e.g. negative ints).
    return bool(np.all((exp == 0) | (exp == 0xFF)))


def _clean_float(x) -> float:
    """float32 -> Python float with the shortest decimal repr (0.39157 not
    0.3915700018405914)."""
    return float(str(np.float32(x)))


def decode_heuristic(b: bytes):
    n = len(b)
    if n == 0:
        return None
    if n == 1:
        return bool(b[0])
    # 4-byte streams are numbers: an int in 32..126 would otherwise look
    # like a one-character string (100 -> 'd').
    if n != 4 and _looks_like_text(b):
        return _cstr(b)
    if n % 4:
        if not b.strip(b"\0"):
            return ""
        if set(b) <= {0, 1}:
            return bools(b)  # packed flag bytes
        return b  # unknown binary blob
    words = np.frombuffer(b, "<u4")
    if not words.any():
        # all zeros: an empty string buffer or a zero-filled array/scalar.
        if n == 4:
            return 0
        return "" if n in _COMMON_STRING_SIZES else np.zeros(n // 4, np.float32)
    if _int_like(words):
        arr = words.view("<i4")
        return int(arr[0]) if n == 4 else arr.copy()
    arr = words.view("<f4")
    return _clean_float(arr[0]) if n == 4 else arr.copy()


# Buffer sizes Xradia uses for fixed-width string fields.
_COMMON_STRING_SIZES = {16, 20, 24, 25, 32, 50, 64, 68, 80, 100, 125, 128,
                        150, 192, 250, 255, 256, 260, 261, 300, 425, 500,
                        512, 640, 1024, 2550}


# --------------------------------------------------------------------------
# explicit decoders
# --------------------------------------------------------------------------
def string_records(b: bytes, width: int) -> list[str]:
    return [_cstr(b[i:i + width]) for i in range(0, len(b) - width + 1, width)]


def parse_xradia_date(s: str) -> datetime | None:
    for fmt in ("%m/%d/%Y %H:%M:%S.%f", "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def f32(b: bytes) -> np.ndarray:
    return np.frombuffer(b[: len(b) // 4 * 4], "<f4").copy()


def i32(b: bytes) -> np.ndarray:
    return np.frombuffer(b[: len(b) // 4 * 4], "<i4").copy()


def bools(b: bytes) -> np.ndarray:
    return np.frombuffer(b, np.uint8).astype(bool)


def scalar_f32(b: bytes):
    return _clean_float(np.frombuffer(b[:4], "<f4")[0]) if len(b) >= 4 else None


def scalar_i32(b: bytes):
    return int(np.frombuffer(b[:4], "<i4")[0]) if len(b) >= 4 else None


# Streams whose type the heuristic would get wrong (or ambiguous zeros).
# Keys are lower-case stream *names* (last path component) or full paths.
FLOAT_SCALARS = {
    "pixelsize", "voltage", "current", "power", "exptime", "exposuretime",
    "storadistance", "dtoradistance", "srctora", "dettora", "rotationangle",
    "centershift", "beamhardening", "globalmin", "globalmax", "anglespan",
    "opticalmagnification", "lensmagnification", "optmag", "campixelsize",
    "camerafinerotation", "camerasensitivity", "bgadjustments",
    "readoutfreq", "absorptionscalefactor", "absorptionscaleoffset",
    "transmissionscalefactor", "reconfiltersmoothfactor", "sourcevoltage",
    "meansamplex", "meansampley", "meansamplez", "netspotoffsetblxum",
    "netspotoffsetblyum", "apertureexitz", "pitch", "yaw", "pitchoptics",
    "yawoptics", "objoffsetx", "objoffsety", "detectoroffset",
    "samplexstitchposition", "sampleystitchposition",
    "samplezstitchposition", "fringestrength", "fringewidth",
    "stddevvalhe", "stddevvalle", "maxconebeamangle", "xshift", "yshift",
}
INT_SCALARS = {
    "datatype", "imagewidth", "imageheight", "noofimages", "imagestaken",
    "camerabinning", "horizontalbin", "verticalalbin", "verticalbin",
    "camfullwidth", "camfullheight", "totalaxis", "filetype",
    "numofprojects", "reconbinning", "binning", "croppedwidth",
    "croppedheight", "mosiacrows", "mosiaccolumns", "mosaicrows",
    "mosaiccolumns", "temperature", "objectiveid", "cameraserialnumber",
}
FLOAT_ARRAYS = {
    "angles", "xposition", "yposition", "zposition", "exptimes", "energy",
    "conangle", "coneangle", "fanangle", "camera temperature",
    "cameratemperature", "storadistance", "dtoradistance", "xraycurrent",
    "xrayvoltage", "xraymagnification", "ionchambercurrent", "sourcespotx",
    "sourcespoty", "thetaparallax", "reconsearchvalue", "focusaxispos",
    "anglevector", "x-shifts", "y-shifts", "conversionfactors",
    "useroffsets", "distortparam",
}
INT_ARRAYS = {
    "framesaveraged", "framesperimage", "imagesperprojection", "hasimage",
    "norecoveryretries", "sourcefilterindex", "selectedimages",
    "indextoxradiaid", "aicapplied",
}
BOOL_ARRAYS = {"hwrestablized", "hwrestabilized"}

# Name fragments of fixed-width string fields.  Needed because an *empty*
# string buffer is often followed by uninitialised memory, which is
# indistinguishable from numbers by content alone.
STRING_HINTS = ("name", "file", "serial", "partnum", "location", "path",
                "units", "model", "comment", "description", "version_str",
                "filter", "string")
_NOT_STRING = {"reconfilter", "reconfilterchoice", "sourcefilterid",
               "sourcefilterindex", "usectscalefilter", "filetype",
               "fileinfover"}


def _is_string_name(name: str) -> bool:
    return name not in _NOT_STRING and any(h in name for h in STRING_HINTS)


def guess_group_size(sizes: dict[str, int]) -> int | None:
    """Guess the table length of a storage from its siblings' sizes.

    Xradia stores tables (one entry per image, per motor axis, per objective,
    ...) as one stream per column, so most array streams in a storage share
    the same element count.  Returns the most common ``n // 4`` among
    word-aligned streams larger than one word, if it occurs at least 3 times.
    """
    counts: dict[int, int] = {}
    for n in sizes.values():
        if n > 4 and n % 4 == 0:
            counts[n // 4] = counts.get(n // 4, 0) + 1
    if not counts:
        return None
    g, c = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    return g if c >= 3 and g >= 2 else None


def _decode_table_column(b: bytes, g: int):
    """Decode a stream that holds ``g`` equal-width records, or None."""
    n = len(b)
    if n % g:
        return None
    w = n // g
    if w == 1:
        return bools(b) if set(b) <= {0, 1} else np.frombuffer(b, np.uint8).copy()
    if w == 4:
        v = decode_heuristic(b)
        return None if isinstance(v, str) and v == "" else v
    recs = [b[i:i + w] for i in range(0, n, w)]
    if any(_looks_like_text(r) for r in recs) and all(
            _looks_like_text(r) or not r.split(b"\0", 1)[0] for r in recs):
        return [_cstr(r) for r in recs]
    if w % 4 == 0:
        words = np.frombuffer(b, "<u4")
        arr = words.view("<i4") if _int_like(words) and words.any() else words.view("<f4")
        return arr.reshape(g, w // 4).copy()
    return None


def decode_stream(path: str, b: bytes, ctx: dict | None = None):
    """Decode stream ``path`` (e.g. 'ImageInfo/Angles') with raw bytes ``b``.

    ``ctx`` may hold:
      * ``n_axes``  - number of motor axes (PositionInfo/TotalAxis)
      * ``group``   - table length of the parent storage
                      (see :func:`guess_group_size`)
    """
    ctx = ctx or {}
    lp = path.lower()
    name = lp.rsplit("/", 1)[-1]
    n = len(b)
    n_ax = ctx.get("n_axes")
    g = ctx.get("group")

    if lp.startswith("positioninfo/") and name in ("axisnames", "axisunits") and n_ax:
        return string_records(b, n // n_ax)
    if lp.startswith("positioninfo/") and "motorpositions" in name:
        a = f32(b)
        if n_ax and a.size % n_ax == 0:
            a = a.reshape(-1, n_ax)
        return a
    if n == 4:
        if name in FLOAT_SCALARS:
            return scalar_f32(b)
        if name in INT_SCALARS:
            return scalar_i32(b)
        return decode_heuristic(b)
    if n > 4:
        if name in BOOL_ARRAYS:
            return bools(b)
        if n % 4 == 0 and name in FLOAT_ARRAYS:
            return f32(b)
        if n % 4 == 0 and name in INT_ARRAYS:
            return i32(b)
        if g and g > 1 and n != 4:
            v = _decode_table_column(b, g)
            if v is not None:
                return v
        if _is_string_name(name) and (n % 4 or n in _COMMON_STRING_SIZES):
            return _cstr(b)
    return decode_heuristic(b)
