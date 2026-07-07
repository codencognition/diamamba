#!/usr/bin/env python3
"""
Build a test set by copying selected .wav and .rttm files into a new directory.

Given:
  - a list file with one filename stem per line (no extension), e.g.
        G0001
        G0002
        ...
  - a source directory containing .wav files
  - a source directory containing .rttm files

This script creates:
    <out-dir>/
        wav/   <- copies of <stem>.wav  for each stem in the list
        rttm/  <- copies of <stem>.rttm for each stem in the list

Usage:
    python build_test_set.py \
        --list test_list.txt \
        --wav-dir /path/to/wavs \
        --rttm-dir /path/to/rttms \
        --out-dir test
"""

import argparse
import os
import shutil
import sys


def read_stems(list_path):
    """Read filename stems from the list file (one per line, blanks/#comments ignored)."""
    stems = []
    seen = set()
    with open(list_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # Defensive: strip a trailing extension if the user included one.
            stem = os.path.splitext(line)[0] if line.lower().endswith((".wav", ".rttm")) else line
            if stem in seen:
                continue
            seen.add(stem)
            stems.append(stem)
    return stems


def copy_one(stem, src_dir, dst_dir, ext):
    """Copy <stem><ext> from src_dir to dst_dir. Returns True on success."""
    src = os.path.join(src_dir, stem + ext)
    if not os.path.isfile(src):
        return False
    shutil.copy2(src, os.path.join(dst_dir, stem + ext))
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Copy a listed subset of .wav and .rttm files into test/wav and test/rttm.")
    ap.add_argument("--list", required=True,
                    help="Text file with one filename stem per line (no extension).")
    ap.add_argument("--wav-dir", required=True,
                    help="Source directory containing .wav files.")
    ap.add_argument("--rttm-dir", required=True,
                    help="Source directory containing .rttm files.")
    ap.add_argument("--out-dir", default="test",
                    help="Output directory to create (default: test).")
    args = ap.parse_args()

    if not os.path.isfile(args.list):
        sys.exit(f"Error: list file not found: {args.list}")
    for d, label in ((args.wav_dir, "wav-dir"), (args.rttm_dir, "rttm-dir")):
        if not os.path.isdir(d):
            sys.exit(f"Error: {label} is not a directory: {d}")

    stems = read_stems(args.list)
    if not stems:
        sys.exit(f"Error: no filenames found in {args.list}")

    wav_out = os.path.join(args.out_dir, "wav")
    rttm_out = os.path.join(args.out_dir, "rttm")
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(rttm_out, exist_ok=True)

    copied_wav = copied_rttm = 0
    missing_wav = []
    missing_rttm = []

    for stem in stems:
        if copy_one(stem, args.wav_dir, wav_out, ".wav"):
            copied_wav += 1
        else:
            missing_wav.append(stem)

        if copy_one(stem, args.rttm_dir, rttm_out, ".rttm"):
            copied_rttm += 1
        else:
            missing_rttm.append(stem)

    print(f"Listed stems:   {len(stems)}")
    print(f"Copied wav:     {copied_wav} -> {wav_out}")
    print(f"Copied rttm:    {copied_rttm} -> {rttm_out}")

    if missing_wav:
        print(f"\n[warn] {len(missing_wav)} wav file(s) not found in {args.wav_dir}:",
              file=sys.stderr)
        for s in missing_wav:
            print(f"  {s}.wav", file=sys.stderr)
    if missing_rttm:
        print(f"\n[warn] {len(missing_rttm)} rttm file(s) not found in {args.rttm_dir}:",
              file=sys.stderr)
        for s in missing_rttm:
            print(f"  {s}.rttm", file=sys.stderr)


if __name__ == "__main__":
    main()