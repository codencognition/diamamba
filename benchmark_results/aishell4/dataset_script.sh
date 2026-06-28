#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
BASE="/workspace/benchmarks/aishell4"
OUT_BASE="$BASE/dataset"
SPLIT_SRC=(train_S train_M train_L)   # split each of these 80/20 into train/dev
TEST_SRC="test"                       # no split, all -> dataset/test
DEV_RATIO="0.20"
SEED="3"
DRY_RUN="${DRY_RUN:-}"                # DRY_RUN=1 to preview, no changes
FORCE="${FORCE:-}"                    # FORCE=1 to skip the confirmation prompt

# outputs
TRAIN_WAV="$OUT_BASE/train/wav";  TRAIN_TG="$OUT_BASE/train/TextGrid"
DEV_WAV="$OUT_BASE/dev/wav";      DEV_TG="$OUT_BASE/dev/TextGrid"
TEST_WAV="$OUT_BASE/test/wav";    TEST_TG="$OUT_BASE/test/TextGrid"

# ---- sanity ----
command -v ffmpeg >/dev/null || { echo "ERROR: ffmpeg not found"; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found"; exit 1; }
for d in "${SPLIT_SRC[@]}" "$TEST_SRC"; do
  [[ -d "$BASE/$d/wav" ]]      || { echo "ERROR: missing $BASE/$d/wav"; exit 1; }
  [[ -d "$BASE/$d/TextGrid" ]] || { echo "ERROR: missing $BASE/$d/TextGrid"; exit 1; }
done

# ---- collision check across split sources (basenames map to same train/dev filenames) ----
names=()
for d in "${SPLIT_SRC[@]}"; do
  while IFS= read -r -d '' f; do names+=("$(basename "${f%.flac}")"); done \
    < <(find "$BASE/$d/wav" -type f -name '*.flac' -print0)
done
dup=$(printf '%s\n' "${names[@]}" | sort | uniq -d)
[[ -n "$dup" ]] && { echo "ERROR: basename collision across ${SPLIT_SRC[*]}:"; echo "$dup"; exit 1; }
echo "No collisions across split sources."

# ---- destructive-action warning ----
if [[ -z "$DRY_RUN" && -z "$FORCE" ]]; then
  echo "WARNING: this will DELETE source .flac and MOVE source TextGrid files out of:"
  for d in "${SPLIT_SRC[@]}" "$TEST_SRC"; do echo "  $BASE/$d"; done
  echo "Sources will be emptied; the script is NOT rerunnable afterwards."
  read -r -p "Type 'yes' to proceed: " ok
  [[ "$ok" == "yes" ]] || { echo "Aborted."; exit 1; }
fi

mkdir -p "$TRAIN_WAV" "$TRAIN_TG" "$DEV_WAV" "$DEV_TG" "$TEST_WAV" "$TEST_TG"

# ---- core: convert one flac, verify, delete flac, move ALL its annotation files ----
# moves every file in the annotation dir matching "<base>.*" (e.g. .TextGrid AND .rttm)
process_one () {
  local flac="$1" tgdir="$2" outwav="$3" outtg="$4"
  local base; base="$(basename "${flac%.flac}")"

  local matches=()
  shopt -s nullglob
  matches=( "$tgdir/$base".* )
  shopt -u nullglob
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "WARN: no annotation files for '$base' in $tgdir -> skipped, flac kept"
    return
  fi

  local dest="$outwav/$base.wav"
  if [[ -n "$DRY_RUN" ]]; then
    echo "DRY: $flac -> $dest ; mv ${#matches[@]} file(s) -> $outtg/  [${matches[*]##*/}]"
    return
  fi
  ffmpeg -nostdin -hide_banner -loglevel error -y \
    -i "$flac" -ac 1 -ar 16000 -sample_fmt s16 "$dest"
  [[ -s "$dest" ]] || { echo "ERROR: conversion failed for $flac (kept)"; exit 1; }
  rm -f "$flac"                  # delete only after wav verified
  mv "${matches[@]}" "$outtg/"   # move all matching annotation files (.TextGrid, .rttm, ...)
}

# ---- split sources: 80% train, 20% dev (stratified, seeded) ----
for src in "${SPLIT_SRC[@]}"; do
  wavdir="$BASE/$src/wav"; tgdir="$BASE/$src/TextGrid"
  echo "== $src : computing split (seed=$SEED, dev=$DEV_RATIO) =="

  if ! dev_raw=$(python3 - "$wavdir" "$DEV_RATIO" "$SEED" <<'PY'
import os, sys, random
d, ratio, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
fl = sorted(f[:-5] for f in os.listdir(d) if f.endswith('.flac'))
random.seed(seed); random.shuffle(fl)
k = max(1, round(len(fl) * ratio)) if fl else 0
print('\n'.join(fl[:k]))
PY
  ); then echo "ERROR: split computation failed for $src"; exit 1; fi

  unset is_dev; declare -A is_dev=()
  dev_bn=()
  [[ -n "$dev_raw" ]] && mapfile -t dev_bn <<< "$dev_raw"
  for b in "${dev_bn[@]}"; do is_dev["$b"]=1; done
  echo "   dev=${#dev_bn[@]} files"

  while IFS= read -r -d '' flac; do
    base="$(basename "${flac%.flac}")"
    if [[ -n "${is_dev[$base]:-}" ]]; then
      process_one "$flac" "$tgdir" "$DEV_WAV"   "$DEV_TG"
    else
      process_one "$flac" "$tgdir" "$TRAIN_WAV" "$TRAIN_TG"
    fi
  done < <(find "$wavdir" -type f -name '*.flac' -print0)
done

# ---- test: no split ----
echo "== $TEST_SRC : converting (no split) =="
while IFS= read -r -d '' flac; do
  process_one "$flac" "$BASE/$TEST_SRC/TextGrid" "$TEST_WAV" "$TEST_TG"
done < <(find "$BASE/$TEST_SRC/wav" -type f -name '*.flac' -print0)

# ---- summary ----
echo "DONE -> $OUT_BASE"
for p in train dev test; do
  echo "  $p: wav=$(find "$OUT_BASE/$p/wav" -type f -name '*.wav' 2>/dev/null | wc -l)  TextGrid=$(find "$OUT_BASE/$p/TextGrid" -type f 2>/dev/null | wc -l)"
done