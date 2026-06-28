import os
import glob
import shutil
import subprocess

WORKSPACE    = "/workspace/benchmarks/voxconverse"
SRC_DIR   = "/workspace/diamamba/src"
CONFIGS_DIR = "/workspace/diamamba/configs"

# ── run name used for output naming (change freely between experiments) ──────
RUN_NAME = "model_noFT"

MODELS_PATH  = "/workspace/models/dr0p15_specaug_on/models"
EPOCHS = "91-100"
SUBSAMP = "10"

# MODELS_PATH  = "/workspace/models/ft_callhome_f1800/models"
# EPOCHS = "441-450"
# SUBSAMP = "10"

MWLEN = "11"
COLLAR = "0.25"

# ── Inputs: pre-prepared audio + reference RTTMs ─────────────────────────────
audio_dir    = f"{WORKSPACE}/test/audio_files"      # contains <meeting_id>.wav
ref_rttm_dir = f"{WORKSPACE}/test/rttm_files"    # contains <meeting_id>.rttm
uem_dir      = None                              # set to a path to enable UEM scoring, or None to skip

output_dir   = f"{WORKSPACE}/hypothesis_rttms_{RUN_NAME}"
combined_hyp = f"{WORKSPACE}/hyp_combined_{RUN_NAME}.rttm"
combined_ref = f"{WORKSPACE}/ref_combined_{RUN_NAME}.rttm"
combined_uem = f"{WORKSPACE}/uem_combined_{RUN_NAME}.uem"

INFER_SCRIPT = f"{SRC_DIR}/infer_single_file.py"
INFER_CONFIG = f"{CONFIGS_DIR}/infer_16k_10attractors.yaml"

# ── Clean up previous run outputs ────────────────────────────────────────────
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"Deleted {output_dir}")
for f in (combined_hyp, combined_ref, combined_uem):
    if os.path.exists(f):
        os.remove(f)
        print(f"Deleted {f}")

os.makedirs(output_dir, exist_ok=True)

# ── Step 1: derive meeting list from audio folder ────────────────────────────
wav_paths = sorted(glob.glob(f"{audio_dir}/*.wav"))
if not wav_paths:
    raise FileNotFoundError(f"No .wav files found in {audio_dir}")
test_meetings = [os.path.splitext(os.path.basename(p))[0] for p in wav_paths]
print(f"Found {len(test_meetings)} recordings in {audio_dir}")

# ── Step 2: concatenate reference RTTMs into one file ────────────────────────
with open(combined_ref, "w") as out_ref:
    for meeting_id in test_meetings:
        rttm_path = f"{ref_rttm_dir}/{meeting_id}.rttm"
        if not os.path.exists(rttm_path):
            raise FileNotFoundError(f"Reference RTTM missing: {rttm_path}")
        with open(rttm_path) as f:
            out_ref.write(f.read())
print(f"Combined reference RTTM: {combined_ref}")

# ── Step 3: combine UEM files into one (optional) ────────────────────────────
use_uem = uem_dir is not None and os.path.isdir(uem_dir)
if use_uem:
    with open(combined_uem, "w") as out_uem:
        for meeting_id in test_meetings:
            uem_path = f"{uem_dir}/{meeting_id}.uem"
            if not os.path.exists(uem_path):
                raise FileNotFoundError(f"UEM missing: {uem_path}")
            with open(uem_path) as f:
                out_uem.write(f.read())
    print(f"Combined UEM ready: {combined_uem}")
else:
    print("No UEM dir set; scoring without UEM")

# ── Step 4: run inference per meeting (single wav mode, no Kaldi needed) ─────
print(f"\nRunning inference: {RUN_NAME}")
for meeting_id in test_meetings:
    print(f"  [{meeting_id}] running inference...")
    subprocess.run([
        "python", INFER_SCRIPT,
        "-c",            INFER_CONFIG,
        "--wav-dir",     audio_dir,
        "--wav-name",    meeting_id,
        "--models-path", MODELS_PATH,
        "--epochs",      EPOCHS,
        "--subsampling", SUBSAMP,
        "--median-window-length", MWLEN,
        "--rttms-dir",   output_dir,
    ], check=True)
    print(f"  [{meeting_id}] done")

# ── Step 5: find and concatenate hypothesis RTTMs into one file ──────────────
rttm_files = glob.glob(f"{output_dir}/**/rttms/*.rttm", recursive=True)
print(f"\nFound {len(rttm_files)} hypothesis RTTMs")

with open(combined_hyp, "w") as outf:
    for rttm in sorted(rttm_files):
        with open(rttm) as inf:
            outf.write(inf.read())
print(f"Combined hypothesis RTTM: {combined_hyp}")

# ── Step 6: score ────────────────────────────────────────────────────────────
print("\nScoring...")
# score_cmd = [
#     "python", f"{WORKSPACE}/dscore/score.py",
#     "-r", combined_ref,
#     "-s", combined_hyp,
#     "--collar", COLLAR,
# ]
log_file = f"{WORKSPACE}/scoring_mamba.log"
score_cmd = (
    f"python {WORKSPACE}/dscore/score.py "
    f"-r {combined_ref} "
    f"-s {combined_hyp} "
    f"--collar {COLLAR} "
    f"2>&1 | tee {log_file}"
)

if use_uem:
    score_cmd += ["-u", combined_uem]
subprocess.run(score_cmd, shell=True, check=True)