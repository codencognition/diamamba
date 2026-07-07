#!/usr/bin/env python3
"""
Generate RTTM files from MAGICDATA Mandarin Chinese Conversational Speech Corpus (RAMC).

Each transcript .txt file contains lines of the form:

    [start,end]<TAB>SPEAKER_ID<TAB>gender,dialect<TAB>transcription

e.g.

    [0.811,6.057]	G00000168	男,普通话	爱数智慧语音采集二零一九年十月二十五日

This script reads each transcript, pairs it with its matching .wav file, and writes a
standard NIST RTTM (Rich Transcription Time Marked) file with one SPEAKER line per segment.

RTTM SPEAKER line format (10 fields, space-separated):

    SPEAKER <file-id> <channel> <onset> <duration> <NA> <NA> <speaker-id> <NA> <NA>

Usage:
    python generate_rttm.py --wav-dir WAV_DIR --txt-dir TXT_DIR --out-dir OUT_DIR
"""

import argparse
import os
import re
import sys

# Speaker ID that denotes non-speech / no speaker in this corpus (无 = "none").
# Segments labelled with this ID contain only noise tags like [*] or [+], so by
# default we drop them. Pass --keep-nonspeech to retain them.
NONSPEECH_SPEAKER_ID = "G00000000"

# Matches a leading time interval like  [0.811,6.057]  with optional inner spaces.
INTERVAL_RE = re.compile(r"^\s*\[\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\]")


def parse_transcript(path):
    """Parse one transcript file into a list of (start, end, speaker_id) tuples."""
    segments = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue

            m = INTERVAL_RE.match(line)
            if not m:
                print(f"  [warn] {os.path.basename(path)}:{lineno}: "
                      f"no time interval found, skipping: {line!r}", file=sys.stderr)
                continue

            start = float(m.group(1))
            end = float(m.group(2))

            # Remaining content after the interval, split on tabs.
            rest = line[m.end():].lstrip("\t ")
            fields = rest.split("\t")
            speaker_id = fields[0].strip() if fields and fields[0].strip() else "UNKNOWN"

            segments.append((start, end, speaker_id))
    return segments


def write_rttm(segments, file_id, out_path, channel="1",
               keep_nonspeech=False, min_duration=0.0):
    """Write segments to an RTTM file. Returns number of lines written."""
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for start, end, speaker_id in segments:
            if not keep_nonspeech and speaker_id == NONSPEECH_SPEAKER_ID:
                continue

            duration = end - start
            if duration <= 0:
                print(f"  [warn] {file_id}: non-positive duration "
                      f"({start} -> {end}) for speaker {speaker_id}, skipping",
                      file=sys.stderr)
                continue
            if duration < min_duration:
                continue

            # SPEAKER <file-id> <chnl> <onset> <dur> <ortho> <stype> <name> <conf> <slat>
            out.write(
                f"SPEAKER {file_id} {channel} "
                f"{start:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n"
            )
            written += 1
    return written


def find_transcript(txt_dir, stem):
    """Find the transcript file matching a wav stem. Tries .txt then .TXT."""
    for ext in (".txt", ".TXT"):
        cand = os.path.join(txt_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Generate RTTM files from MAGICDATA RAMC transcripts.")
    ap.add_argument("--wav-dir", required=True,
                    help="Directory containing the .wav files.")
    ap.add_argument("--txt-dir", required=True,
                    help="Directory containing the corresponding .txt transcripts.")
    ap.add_argument("--out-dir", required=True,
                    help="Directory where .rttm files will be written.")
    ap.add_argument("--channel", default="1",
                    help="Channel field for RTTM lines (default: 1).")
    ap.add_argument("--keep-nonspeech", action="store_true",
                    help=f"Keep segments labelled {NONSPEECH_SPEAKER_ID} "
                         "(non-speech). Dropped by default.")
    ap.add_argument("--min-duration", type=float, default=0.0,
                    help="Drop segments shorter than this many seconds "
                         "(default: 0.0, keep all).")
    ap.add_argument("--match-by", choices=["wav", "txt"], default="wav",
                    help="Iterate over .wav files (default) or .txt files to "
                         "decide which RTTMs to produce.")
    args = ap.parse_args()

    for d, label in ((args.wav_dir, "wav-dir"), (args.txt_dir, "txt-dir")):
        if not os.path.isdir(d):
            sys.exit(f"Error: {label} is not a directory: {d}")

    os.makedirs(args.out_dir, exist_ok=True)

    total_files = 0
    total_segments = 0
    missing = 0

    if args.match_by == "wav":
        names = sorted(n for n in os.listdir(args.wav_dir)
                       if n.lower().endswith(".wav"))
        if not names:
            sys.exit(f"Error: no .wav files found in {args.wav_dir}")

        for name in names:
            stem = os.path.splitext(name)[0]
            txt_path = find_transcript(args.txt_dir, stem)
            if txt_path is None:
                print(f"[skip] {name}: no matching transcript in {args.txt_dir}",
                      file=sys.stderr)
                missing += 1
                continue

            segments = parse_transcript(txt_path)
            out_path = os.path.join(args.out_dir, stem + ".rttm")
            n = write_rttm(segments, stem, out_path,
                           channel=args.channel,
                           keep_nonspeech=args.keep_nonspeech,
                           min_duration=args.min_duration)
            print(f"[ok] {name} -> {stem}.rttm ({n} segments)")
            total_files += 1
            total_segments += n
    else:
        names = sorted(n for n in os.listdir(args.txt_dir)
                       if n.lower().endswith(".txt"))
        if not names:
            sys.exit(f"Error: no .txt files found in {args.txt_dir}")

        for name in names:
            stem = os.path.splitext(name)[0]
            wav_path = os.path.join(args.wav_dir, stem + ".wav")
            if not os.path.isfile(wav_path):
                print(f"[note] {name}: no matching .wav in {args.wav_dir} "
                      "(writing RTTM anyway)", file=sys.stderr)

            txt_path = os.path.join(args.txt_dir, name)
            segments = parse_transcript(txt_path)
            out_path = os.path.join(args.out_dir, stem + ".rttm")
            n = write_rttm(segments, stem, out_path,
                           channel=args.channel,
                           keep_nonspeech=args.keep_nonspeech,
                           min_duration=args.min_duration)
            print(f"[ok] {name} -> {stem}.rttm ({n} segments)")
            total_files += 1
            total_segments += n

    print(f"\nDone. Wrote {total_files} RTTM file(s), "
          f"{total_segments} total segment(s), to {args.out_dir}")
    if missing:
        print(f"{missing} wav file(s) had no matching transcript.",
              file=sys.stderr)


if __name__ == "__main__":
    main()