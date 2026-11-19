"""Command line tools.

``xradia-info FILE [--json OUT] [--tiff OUT] [--tree] [--grep REGEX]``
    FILE may be an Xradia .txm/.txrm/.xrm or an .h5 archive.
``xradia-archive FILES... [--out-dir DIR] [--drop-bits N|auto] [--verify-only]``
    Compress Xradia files to HDF5 archives (lossless by default).
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from pathlib import Path

from .archive import open_any, verify_hdf5, write_hdf5
from .reader import XradiaFile, _json_default


def _walk(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _walk(v, f"{prefix}{k}/")
        else:
            yield prefix + k, v


def _short(v, width=100):
    if isinstance(v, np.ndarray):
        with np.printoptions(threshold=6, edgeitems=3, linewidth=10**6, precision=6):
            s = f"{v.dtype}{list(v.shape)} {v}"
    elif isinstance(v, list):
        s = f"list[{len(v)}] {v[:3]}{' ...' if len(v) > 3 else ''}"
    elif isinstance(v, bytes):
        s = f"bytes[{len(v)}] {v[:16].hex()}..."
    else:
        s = repr(v)
    s = s.replace("\n", " ")
    return s if len(s) <= width else s[: width - 3] + "..."


def main(argv=None):
    ap = argparse.ArgumentParser(prog="xradia-info",
                                 description="Show metadata of / export an Xradia file or archive.")
    ap.add_argument("file")
    ap.add_argument("--json", metavar="OUT", help="write summary + full metadata as JSON")
    ap.add_argument("--tiff", metavar="OUT", help="write ImageJ-calibrated TIFF stack (data unchanged)")
    ap.add_argument("--per-slice", action="store_true", help="with --tiff: one file per image")
    ap.add_argument("--tree", action="store_true", help="list every decoded stream")
    ap.add_argument("--grep", metavar="REGEX", help="only streams whose path matches (implies --tree)")
    ap.add_argument("--no-config", action="store_true", help="hide ConfigureBackup/* in --tree")
    a = ap.parse_args(argv)

    with open_any(a.file) as f:
        if a.json:
            f.to_json(a.json)
            print(f"wrote {a.json}", file=sys.stderr)
        if a.tiff:
            out = f.to_tiff(a.tiff, per_slice=a.per_slice)
            print(f"wrote {len(out)} file(s): {out[0]}{' ...' if len(out) > 1 else ''}",
                  file=sys.stderr)
        if a.tree or a.grep:
            rx = re.compile(a.grep, re.I) if a.grep else None
            for path, v in _walk(f.metadata):
                if a.no_config and path.startswith("ConfigureBackup/"):
                    continue
                if rx and not rx.search(path):
                    continue
                print(f"{path:70s} {_short(v)}")
        elif not (a.json or a.tiff):
            print(repr(f))
            print(json.dumps(f.summary, indent=2, default=_json_default))


def _mb(n):
    return f"{n / 1e6:,.0f} MB"


def archive_main(argv=None):
    ap = argparse.ArgumentParser(
        prog="xradia-archive",
        description="Compress Xradia files to HDF5 (gzip+shuffle, readable by any HDF5 tool).")
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--out-dir", help="output directory (default: next to each input)")
    ap.add_argument("--drop-bits", default="0",
                    help="0 = lossless (default); N or 'auto' rounds to multiples of 2**N")
    ap.add_argument("--level", type=int, default=4, help="gzip level 1-9 (default 4)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the read-back check")
    ap.add_argument("--verify-only", action="store_true",
                    help="FILES are existing .h5 archives: re-check their pixel checksums")
    a = ap.parse_args(argv)

    if a.verify_only:
        bad = 0
        for fn in a.files:
            r = verify_hdf5(fn)
            bad += not r["ok"]
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {fn} {r}")
        sys.exit(1 if bad else 0)

    drop = a.drop_bits if a.drop_bits == "auto" else int(a.drop_bits)
    tot_in = tot_out = 0
    failed = 0
    for fn in a.files:
        src = Path(fn)
        dst = (Path(a.out_dir) if a.out_dir else src.parent) / (src.stem + ".h5")
        if dst.exists() and not a.overwrite:
            print(f"skip {src} -> {dst} exists (use --overwrite)", file=sys.stderr)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)

        def progress(i, n):
            print(f"\r  {src.name}: {i}/{n}", end="", file=sys.stderr, flush=True)

        with XradiaFile(src) as xf:
            r = write_hdf5(xf, dst, drop_bits=drop, level=a.level,
                           verify=not a.no_verify, progress=progress)
        print("\r", end="", file=sys.stderr)
        tot_in += r["source_bytes"]
        tot_out += r["archive_bytes"]
        ok = r.get("verified", True)
        failed += not ok
        loss = "lossless" if r["drop_bits"] == 0 else \
            f"drop {r['drop_bits']} bits (max change {r['max_abs_change']}, noise {r['noise_sigma']:.0f})"
        print(f"{'OK  ' if ok else 'FAIL'} {src.name} -> {dst}: {_mb(r['source_bytes'])} -> "
              f"{_mb(r['archive_bytes'])} ({r['ratio']:.2f}x), {loss}, "
              f"{r['write_s']} s" + (f" + verify {r['verify_s']} s" if "verify_s" in r else ""))
    if tot_out:
        print(f"total {_mb(tot_in)} -> {_mb(tot_out)} ({tot_in / tot_out:.2f}x)")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
