"""
Mix AMI per-speaker headset wavs into a single mixed wav per meeting,
and prepare matching UEM / RTTM / list files for diarization training.

Key fixes vs. the original script:
  * Mix by SUM + peak-normalize instead of MEAN
      (mean attenuates speech by ~1/N and causes log-mel underflow → NaN downstream).
  * Verify sample-rate consistency across all headsets in a meeting.
  * Warn loudly on stereo headset files and large length spreads.
  * Write FLOAT wavs (preserve dynamic range; loader will cast to float anyway).
  * Stronger NaN/Inf/silence checks before writing.
"""

import os
import glob
import shutil
import numpy as np
import soundfile as sf
from collections import defaultdict

WORKSPACE = "/workspace/benchmarks/amiheadset"
audio_dir = f"{WORKSPACE}/dataset"
mixed_dir = f"{WORKSPACE}/mixed_wavs"
AMI_SETUP = f"{WORKSPACE}/AMI-diarization-setup"

# Output wav format. 'FLOAT' preserves dynamic range; switch to 'PCM_24' if your
# training pipeline insists on integer PCM. Avoid PCM_16 for mixed multi-speaker
# tracks — quantization in quiet segments hurts diarization.
OUTPUT_SUBTYPE = "PCM_16"

# Headroom after peak-normalization (0.95 → -0.45 dBFS peak)
NORMALIZE_PEAK = 0.95

os.makedirs(mixed_dir, exist_ok=True)


def mix_headsets(meeting_id: str, headset_paths: list) -> tuple:
    """Sum all headset tracks and peak-normalize. Returns (mixed, sr)."""
    arrays, sr = [], None

    for path in headset_paths:
        audio, rate = sf.read(path, always_2d=False)

        # Sample-rate consistency check
        if sr is None:
            sr = rate
        elif rate != sr:
            raise ValueError(
                f"[{meeting_id}] sample-rate mismatch: {rate} vs {sr} in {path}"
            )

        # Headset tracks should be mono — flag if not
        if audio.ndim == 2:
            print(f"    [{meeting_id}] WARNING: {os.path.basename(path)} is stereo "
                  f"({audio.shape[1]}ch); averaging channels — verify this is expected")
            audio = audio.mean(axis=1)

        # Reject any input that already has bad values
        if not np.all(np.isfinite(audio)):
            raise ValueError(
                f"[{meeting_id}] non-finite values in input {os.path.basename(path)}"
            )

        arrays.append(audio.astype(np.float64))  # accumulate in float64 to avoid overflow

    # Length spread diagnostic — large spread means something is wrong upstream
    lens = [len(a) for a in arrays]
    if max(lens) - min(lens) > sr:  # > 1 second
        print(f"  [{meeting_id}] WARNING: headset length spread = "
              f"{max(lens) - min(lens)} samples ({(max(lens) - min(lens)) / sr:.2f}s)")

    # Zero-pad to common length, then SUM
    max_len = max(lens)
    arrays = [np.pad(a, (0, max_len - len(a))) for a in arrays]
    mixed = np.sum(np.stack(arrays), axis=0)

    # Peak-normalize (with headroom)
    peak = float(np.max(np.abs(mixed)))
    if peak > 0:
        mixed = mixed / peak * NORMALIZE_PEAK
    else:
        print(f"  [{meeting_id}] WARNING: mixed signal is entirely silent")

    # Sanity checks
    assert mixed.ndim == 1, f"[{meeting_id}] mixed audio is not mono: {mixed.shape}"
    assert np.all(np.isfinite(mixed)), f"[{meeting_id}] mixed audio has NaN/Inf"
    rms = float(np.sqrt(np.mean(mixed ** 2)))
    if rms < 1e-4:
        print(f"  [{meeting_id}] WARNING: very low RMS after mix ({rms:.2e}) — "
              f"check input headsets")

    return mixed.astype(np.float32), sr

for split in ["train", "dev", "test"]:
    print(f"\n{'=' * 60}")
    print(f"Processing split: {split}")
    print(f"{'=' * 60}")

    # ── Read meeting list ──────────────────────────────────────────────────
    meetings_file = f"{AMI_SETUP}/lists/{split}.meetings.txt"
    meetings = [m.strip() for m in open(meetings_file).read().splitlines() if m.strip()]
    print(f"[{split}] {len(meetings)} meetings found")

    # ── Step 1: Mix headsets ───────────────────────────────────────────────
    for meeting_id in meetings:
        out_path = f"{mixed_dir}/{meeting_id}.wav"
        if os.path.exists(out_path):
            print(f"  [{meeting_id}] already mixed, skipping")
            continue

        headset_paths = sorted(glob.glob(
            f"{audio_dir}/{meeting_id}/audio/{meeting_id}.Headset-*.wav"
        ))
        if not headset_paths:
            print(f"  [{meeting_id}] WARNING: no headset wavs found, skipping")
            continue

        try:
            mixed, sr = mix_headsets(meeting_id, headset_paths)
        except Exception as e:
            print(f"  [{meeting_id}] ERROR: {e}")
            continue

        sf.write(out_path, mixed, sr, subtype=OUTPUT_SUBTYPE)
        print(f"  [{meeting_id}] mixed -> {out_path}  "
              f"({sr} Hz, {len(mixed) / sr:.1f}s, mono, "
              f"peak={np.max(np.abs(mixed)):.3f}, {len(headset_paths)} headsets)")

    # ── Step 2: Combine UEM files ──────────────────────────────────────────
    combined_uem = f"{WORKSPACE}/{split}_combined.uem"
    uem_dir = f"{AMI_SETUP}/uems/{split}"
    missing_uem = []

    with open(combined_uem, "w") as out_uem:
        for meeting_id in meetings:
            uem_path = f"{uem_dir}/{meeting_id}.uem"
            if os.path.exists(uem_path):
                with open(uem_path) as f:
                    out_uem.write(f.read())
            else:
                missing_uem.append(meeting_id)

    print(f"  Combined UEM     -> {combined_uem}")
    if missing_uem:
        print(f"  WARNING: missing UEM for: {missing_uem}")

    # ── Step 3: Split reference RTTM into per-meeting files ────────────────
    rttm_out_dir = f"{WORKSPACE}/{split}_rttms"
    os.makedirs(rttm_out_dir, exist_ok=True)

    rttm_src = f"{AMI_SETUP}/only_words/rttms/{split}.rttm"

    if os.path.exists(rttm_src):
        bucketed = defaultdict(list)
        with open(rttm_src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                meeting_id = line.split()[1]
                bucketed[meeting_id].append(line)

        missing_rttm = []
        for meeting_id in meetings:
            if meeting_id not in bucketed:
                missing_rttm.append(meeting_id)
                continue
            out_rttm = f"{rttm_out_dir}/{meeting_id}.rttm"
            with open(out_rttm, "w") as f:
                f.write("\n".join(bucketed[meeting_id]) + "\n")

        print(f"  Per-meeting RTTMs -> {rttm_out_dir}/  ({len(bucketed)} files)")
        if missing_rttm:
            print(f"  WARNING: no RTTM entries for: {missing_rttm}")

    else:
        # Fallback: look for pre-split per-meeting RTTMs and copy them
        rttm_dir = f"{AMI_SETUP}/only_words/rttms/{split}"
        missing_rttm = []
        for meeting_id in meetings:
            src = f"{rttm_dir}/{meeting_id}.rttm"
            dst = f"{rttm_out_dir}/{meeting_id}.rttm"
            if os.path.exists(src):
                shutil.copy(src, dst)
            else:
                missing_rttm.append(meeting_id)
        print(f"  Per-meeting RTTMs -> {rttm_out_dir}/")
        if missing_rttm:
            print(f"  WARNING: missing RTTM for: {missing_rttm}")

    # ── Step 4: Write wav name list (no extension) ─────────────────────────
    list_path = f"{WORKSPACE}/list_{split}"
    with open(list_path, "w") as f:
        f.write("\n".join(meetings) + "\n")
    print(f"  Wav name list    -> {list_path}")


print("\nDone. Outputs:")
for split in ["train", "dev", "test"]:
    print(f"  [{split}] UEM      : {WORKSPACE}/{split}_combined.uem")
    print(f"  [{split}] RTTMs    : {WORKSPACE}/{split}_rttms/")
    print(f"  [{split}] wav list : {WORKSPACE}/list_{split}")
print(f"  [all]   wavs     : {mixed_dir}/")

# ── Copy only test wavs listed in list_test ────────────────────────────────
test_wavs_dir = f"{WORKSPACE}/test_wavs"
list_test_path = f"{WORKSPACE}/list_test"

os.makedirs(test_wavs_dir, exist_ok=True)

with open(list_test_path) as f:
    test_meetings = [line.strip() for line in f if line.strip()]

missing_test_wavs = []

for meeting_id in test_meetings:
    src = f"{mixed_dir}/{meeting_id}.wav"
    dst = f"{test_wavs_dir}/{meeting_id}.wav"

    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        missing_test_wavs.append(meeting_id)

print(f"  [test]  test wavs : {test_wavs_dir}/  ({len(test_meetings) - len(missing_test_wavs)} files copied)")

if missing_test_wavs:
    print(f"  WARNING: missing test wavs for: {missing_test_wavs}")