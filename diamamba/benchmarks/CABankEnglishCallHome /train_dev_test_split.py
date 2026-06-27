#!/usr/bin/env python3
"""
python3 train_dev_test_split.py --seed 3

Split TalkBank CallHome audio (.wav) + label (.rttm) files into
train / dev / test sets.

    train.list
    dev.list
    test.list
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

AUDIO_EXTS = {".wav"}
RTTM_EXT = ".rttm"

DEFAULT_BASE = "/workspace/benchmarks/talkbank_callhome"
DEFAULT_AUDIO = f"{DEFAULT_BASE}/dataset/audio_files"
DEFAULT_RTTM = f"{DEFAULT_BASE}/dataset/rttm_files"
DEFAULT_OUTPUT = DEFAULT_BASE


def find_pairs(audio_dir: Path, rttm_dir: Path):
    """Match wav <-> rttm by filename stem."""
    audio = {
        p.stem: p
        for p in sorted(audio_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS
    }

    rttm = {
        p.stem: p
        for p in sorted(rttm_dir.iterdir())
        if p.is_file() and p.suffix.lower() == RTTM_EXT
    }

    common = sorted(set(audio) & set(rttm))
    orphan_audio = sorted(set(audio) - set(rttm))
    orphan_rttm = sorted(set(rttm) - set(audio))

    pairs = [(stem, audio[stem], rttm[stem]) for stem in common]

    return pairs, orphan_audio, orphan_rttm


def two_stage_split(items, seed, holdout_frac):
    """
    80 / 10 / 10 split by default.

    stage 1: 80% train, 20% holdout
    stage 2: holdout split into 50% dev and 50% test
    """
    items = list(items)

    rng = random.Random(seed)
    rng.shuffle(items)

    n = len(items)
    n_holdout = round(n * holdout_frac)
    n_train = n - n_holdout

    train = items[:n_train]
    holdout = items[n_train:]

    n_dev = round(len(holdout) * 0.5)

    dev = holdout[:n_dev]
    test = holdout[n_dev:]

    return train, dev, test


def place(pairs, split_dir: Path, link: bool):
    """Copy or symlink wav and rttm files into split folders."""
    audio_out = split_dir / "audio_files"
    rttm_out = split_dir / "rttm_files"

    audio_out.mkdir(parents=True, exist_ok=True)
    rttm_out.mkdir(parents=True, exist_ok=True)

    for _stem, wav, rttm in pairs:
        a_dst = audio_out / wav.name
        r_dst = rttm_out / rttm.name

        if link:
            for src, dst in ((wav, a_dst), (rttm, r_dst)):
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
        else:
            shutil.copy2(wav, a_dst)
            shutil.copy2(rttm, r_dst)


def write_list_file(pairs, list_path: Path):
    """
    Write only audio filenames into .list file.

    Example:
        iaaa.wav
        iabc.wav
    """
    lines = [wav.stem for _stem, wav, _rttm in pairs]
    list_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Split CallHome wav + rttm files into train/dev/test and create train.list/dev.list/test.list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument("--audio-dir", default=DEFAULT_AUDIO, help="Directory of .wav files")
    ap.add_argument("--rttm-dir", default=DEFAULT_RTTM, help="Directory of .rttm files")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT, help="Where train/dev/test folders and list files are created")
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction held out after stage 1. 0.2 means 80/10/10 split",
    )
    ap.add_argument("--link", action="store_true", help="Symlink instead of copying files")
    ap.add_argument("--dry-run", action="store_true", help="Print split but do not write files")

    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    rttm_dir = Path(args.rttm_dir)
    output_dir = Path(args.output_dir)

    for d in (audio_dir, rttm_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: directory not found: {d}")

    pairs, orphan_audio, orphan_rttm = find_pairs(audio_dir, rttm_dir)

    if orphan_audio:
        print(
            f"WARNING: {len(orphan_audio)} .wav file(s) have no matching .rttm: "
            f"{orphan_audio[:5]}{' ...' if len(orphan_audio) > 5 else ''}"
        )

    if orphan_rttm:
        print(
            f"WARNING: {len(orphan_rttm)} .rttm file(s) have no matching .wav: "
            f"{orphan_rttm[:5]}{' ...' if len(orphan_rttm) > 5 else ''}"
        )

    if not pairs:
        sys.exit("ERROR: no matched wav/rttm pairs found.")

    train, dev, test = two_stage_split(
        pairs,
        seed=args.seed,
        holdout_frac=args.holdout_frac,
    )

    print(f"\nMatched pairs: {len(pairs)}")
    print(f"  train: {len(train)}")
    print(f"  dev  : {len(dev)}")
    print(f"  test : {len(test)}")
    print(f"  seed : {args.seed}")
    print(f"  holdout_frac: {args.holdout_frac}\n")

    if args.dry_run:
        print("(dry run -- no files written)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    place(train, output_dir / "train", args.link)
    place(dev, output_dir / "dev", args.link)
    place(test, output_dir / "test", args.link)

    write_list_file(train, output_dir / "train.list")
    write_list_file(dev, output_dir / "dev.list")
    write_list_file(test, output_dir / "test.list")

    action = "Symlinked" if args.link else "Copied"

    print(f"{action} files into:")
    print(f"  {output_dir / 'train'}")
    print(f"  {output_dir / 'dev'}")
    print(f"  {output_dir / 'test'}")

    print("\nList files created:")
    print(f"  {output_dir / 'train.list'}")
    print(f"  {output_dir / 'dev.list'}")
    print(f"  {output_dir / 'test.list'}")


if __name__ == "__main__":
    main()