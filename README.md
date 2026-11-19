# xradia-reader

Small reader for Zeiss Xradia files (Versa 620 and similar):
`.xrm` (single image / mosaic), `.txrm` (projection series) and `.txm`
(reconstructed volume). Depends only on `numpy` and `olefile`.

Unlike `dxchange.read_txrm`, which reads about 15 fields, this decodes **every**
metadata stream in the file (about 300 fields outside `ConfigureBackup` in the sample `.txm`, plus
the 1629-stream instrument configuration snapshot) and builds a curated
summary with units and derived quantities.

```python
from xradia_reader import XradiaFile

with XradiaFile("scan.txm") as f:
    print(f)                       # kind, n_images, shape, dtype
    sl   = f[500]                  # one image/slice, (height, width)
    sub  = f[100:200, 300:700]     # only the needed streams are read
    vol  = f.read(dtype="float32") # whole stack; out=np.memmap(...) for huge files
    mu   = f.read(slice(0, 10), scaled=True)  # uint16 -> GlobalMin..GlobalMax
    f.summary["geometry"]          # curated, with units
    f.metadata["ImageInfo"]["Angles"]          # full decoded tree
    f.metadata["PositionInfo"]["MotorPositions"]  # (n_images, n_axes)
    f.to_json("scan_metadata.json")
    f.to_tiff("scan.tif")          # Fiji-calibrated stack, data unchanged
```

`to_tiff` writes the stored pixels untouched (uint16 for Versa reconstructions,
the same values `dxchange.read_txm` returns) as an ImageJ hyperstack with the
voxel size: XY resolution tags, `unit=um` and Z `spacing` (volumes only). The
summary goes in ImageJ's Info property (*Image › Show Info*). Slices are
streamed, so the volume is never held in memory. `per_slice=True` writes one
calibrated file per image instead. dxchange's `write_tiff` stores no
calibration at all.

Command line (`pip install -e .` provides `xradia-info` and `xradia-archive`, or use
`python -m xradia_reader.cli`):

```
xradia-info scan.txm                      # summary
xradia-info scan.txm --tree --no-config   # every decoded stream
xradia-info scan.txm --grep 'Recon|Filter'
xradia-info scan.txm --json meta.json
xradia-info scan.txm --tiff scan.tif      # add --per-slice for one file per image
```

## How decoding works

Each metadata item is a separate OLE stream of raw little-endian bytes, with no
type information. `xradia_reader.decode` infers types:

* known layouts: `Date` (40-byte strings per image), `PositionInfo` axis
  names/units (fixed width per axis), motor tables `(n_images, n_axes)`;
* **table columns**: in a storage whose streams mostly hold `G` elements
  (images, motor axes, objectives, …), a stream of `G·w` bytes is split
  into `G` records (bools, numbers, fixed-width strings or rows);
* heuristics: 32-bit words are int32 when all of them look like small or
  negative integers when read as float32 (denormal/NaN), otherwise float32;
  text is detected by content, plus name hints (`*Name`, `*File*`, …) for
  empty string buffers that are followed by uninitialised memory.

Known limitations:

* A value of zero is ambiguous: scalars come back as `0`, arrays as float32.
* 34 `ConfigSources/*Variable` I/O structs stay as raw `bytes`.
* Enum codes (`AcquisitionMode`, `FileType`, `ReconFilterChoice`, …) are
  reported as raw integers. Only `DataType` (5 = uint16, 10 = float32) is
  decoded.
* The `scaled=True` mapping is inferred, not documented by Zeiss: it maps 0…65535 linearly onto
  GlobalMin…GlobalMax, and air comes out at about 0 in the sample.
* Only one `.txm` was available for testing. `.txrm`/`.xrm` paths
  (`ReferenceData`, `Alignment`, `SampleInfo`, mosaics) follow dxchange's
  layout but are untested.

## Archiving to compressed HDF5

```
xradia-archive scans/*.txm -o archive/              # lossless (default)
xradia-archive scans/*.txm -o archive/ --drop-bits auto
xradia-archive --verify-only archive/*.h5           # re-check checksums later
```

Each `.txm` becomes one `.h5` file holding:

- **Pixels:** `/data`, in their original dtype.
- **Voxel size:** `element_size_um` [z, y, x] attribute on `/data`.
- **Metadata:** the full decoded tree in `/metadata`, plus the raw bytes of every
  OLE metadata stream in `/raw_streams`, so nothing depends on the decoding
  heuristics.
- **Summary:** `/summary` as JSON.
- **Checksums:** SHA-256 of the pixels and of the original file.

Every slice is read back and checked after writing.

Compression uses only the HDF5 **shuffle + gzip** filters, which every HDF5
reader decodes without plugins: h5py, MATLAB `h5read`, HDFView, `h5dump`,
Fiji's HDF5 import, Avizo, Dragonfly.
Opening it without this package:

```python
import h5py
with h5py.File("scan.h5") as f:
    vol = f["data"][:]                     # or f["data"][500] for one slice
    dz, dy, dx = f["data"].attrs["element_size_um"]
```
```matlab
vol = h5read('scan.h5', '/data');          % MATLAB reverses axes: (x, y, z)
```

With this package, `open_any("scan.h5")` gives the same API as `XradiaFile`,
including `to_tiff()` for the Fiji workflow.

Results on the sample (2.08 GB, 1014 × 1024 × 1002 uint16, 8 threads):

| mode | size | ratio | time (write + verify) |
|---|---|---|---|
| lossless | 1.32 GB | 1.58× | 16 s + 10 s |
| `--drop-bits auto` (5 bits, max change ±16) | 0.90 GB | 2.31× | 22 s + 10 s |

**Lossless is capped at about 1.6×.** The reconstruction noise
(σ ≈ 900–1900 grey levels) fills about 11 of the 16 bits, and random bits can't
be compressed. Better codecs (zstd, JPEG-LS-type predictors) were measured at
1.61–1.64×, so they aren't worth an exotic dependency.

**`--drop-bits` rounds values to multiples of 2^k** before compression. `auto`
picks the largest k with 2^k ≤ σ_min/16, using the lowest noise found in 10
sampled slices. At that level the noise is far larger than the rounding step,
so the noise breaks up any banding:

- the added error is about 1/50 of the existing noise;
- the total noise rises by about 0.01%;
- averages aren't biased;
- thresholds, segmentation and contrast behave the same.

The one visible effect is a comb-shaped histogram at full 65536-bin resolution.
The data is no longer bit-identical to the instrument file. The setting and the
largest change are recorded in the attributes of `/data`.

## Tests

`python -m pytest tests` (15 tests; the sample-file tests are skipped when
`sample_data/` is missing; the data are git-ignored).
