# DiaMamba

**End-to-end neural speaker diarization with a bidirectional-Mamba frame encoder and a Perceiver-based attractor decoder.**

DiaMamba is an encoder–decoder end-to-end neural diarization (EEND) system that uses a stack of **Bidirectional Mamba (BiMamba)** layers to encode frame embeddings and a Perceiver architecture for attractor-based decoding, giving linear-complexity temporal modeling while retaining the attention-like properties of a transformer-based EEND system. This repository contains the reference implementation, configuration files, and pretrained checkpoints accompanying the paper.

---

## Overview

Modern EEND systems build their frame encoder almost exclusively from self-attention, whose time and memory cost grows **quadratically** with sequence length, bounding the recording duration and speaker count that can be processed in practice. DiaMamba addresses this by using a **pure BiMamba encoder** (four blocks, no self-attention), which is **linear** in sequence length. This makes a long-context training recipe (six-minute training sequences) feasible on commodity hardware and yields large efficiency gains on long recordings, while matching or surpassing the DiaPer baseline in accuracy after in-domain fine-tuning.

DiaMamba formulates diarization as frame-wise multi-label classification. Given `T` frames it outputs per-frame, per-speaker posteriors `Ŷ ∈ [0,1]^{T×A}`, where `A = 10` is the fixed maximum number of decoded attractors; overlapping speech is handled inherently and the number of active speakers is determined from attractor existence probabilities.

**Architecture:**

- **Feature front end:** 40-dim log-Mel filterbanks from 16 kHz audio (25 ms window / 10 ms shift), mean-normalized, stacked with ±7 context frames (600-dim), subsampled ×10 (→ 10 fps).
- **Linear projection + LayerNorm** to encoder dimension `D = 128`.
- **BiMamba frame encoder:** 4 pre-norm residual blocks, each a bidirectional Mamba (forward + time-reversed) with `d_state = 16`, `d_conv = 4`, expansion factor 4, followed by a 512-unit position-wise feed-forward with dropout 0.15. The encoder is conditioned at every layer by intermediate attractors from the shared decoder.
- **Perceiver attractor decoder:** 128 learnable latent vectors, 3 Perceiver blocks (cross-attention + 2 self-attention layers each), producing `A = 10` attractors via a softmax-weighted latent combination plus a per-attractor existence probability.
- **Output layer:** sigmoid of the frame-embedding · attractor dot product.
- ~**4.35 M** trainable parameters.

## Key contributions

1. **First BiMamba encoder within EEND.** The first application of a bidirectional selective state-space encoder inside an end-to-end diarization framework. After in-domain fine-tuning, DiaMamba surpasses the published DiaPer baseline on every corpus with adaptation data.
2. **Substantial efficiency gains.** Large reductions in FLOPs, peak GPU memory, and inference time relative to the self-attention baseline, with an advantage that grows with sequence length (consistent with linear-vs-quadratic scaling).
3. **Reproducible release.** Code and models trained entirely on public, freely available data, with the data pipeline, feature extraction, and scoring held identical to DiaPer to isolate the effect of the encoder.

---

## Repository structure

```
diamamba/
├── README.md
├── LICENSE                         # MIT
├── diamamba/
│   ├── src/
│   │   ├── train.py                # training (3 phases, single entry point)
│   │   ├── infer.py                # inference over a Kaldi data dir → RTTMs
│   │   ├── infer_single_file.py    # inference on a single .wav (no Kaldi needed)
│   │   ├── infer_single_file_gpu_time.py  # single-file inference + GPU timing
│   │   ├── process_data.py         # precompute & cache features (matches configs)
│   │   ├── precompute_features.py  # alternative feature-caching script
│   │   ├── backend/
│   │   │   ├── models.py           # AttractorPerceiver + BidirectionalMamba
│   │   │   ├── losses.py           # PIT diarization / existence / entropy losses
│   │   │   └── updater.py          # optimizers (Noam / Adam) and LR schedule
│   │   └── common_utils/
│   │       ├── diarization_dataset.py   # KaldiDiarizationDataset, PrecomputedDiarizationDataset
│   │       ├── features.py         # STFT, log-Mel, splicing, subsampling
│   │       ├── kaldi_data.py       # Kaldi-style data-dir reader
│   │       ├── metrics.py          # DER and component metrics
│   │       └── gpu_utils.py
│   ├── configs/                    # YAML configs for every stage (see below)
│   └── scripts/
│       ├── prepare_data_dir.sh     # build a Kaldi data dir from wav + rttm
│       └── prepare_data_post_processing.sh  # rebuild utt2spk / spk2utt
├── models/                         # pretrained checkpoints (10 epochs each, for averaging)
│   ├── simulatedConversationLibriSpeechTrain2Spks/   # Phase 1 (epochs 191–200)
│   ├── simulatedConversationLibriSpeechAdapt10Spks/  # Phase 2 (epochs 91–100)
│   ├── amiheadset/                 # Phase 3 fine-tuned (epochs 141–150)
│   ├── aishell4/                   # Phase 3 fine-tuned (epochs 341–350)
│   ├── ramc/                       # Phase 3 fine-tuned (epochs 1041–1050)
│   └── cabanksenglishcallhome/     # Phase 3 fine-tuned (epochs 641–650)
└── benchmarks/                     # per-corpus scoring scripts, split lists, hypothesis RTTMs, dscore logs
    ├── CABankEnglishCallHome/
    ├── aishell4/
    ├── amiheadset/
    ├── ramc/
    └── voxconverse/
```

---

## Requirements

**Hardware.** A CUDA-capable NVIDIA GPU is required — the `mamba-ssm` and `causal-conv1d` kernels are CUDA-only in practice. Development used an **NVIDIA RTX 5070 Ti (~16 GB)**; the six-minute-sequence fine-tuning in the paper additionally used an **NVIDIA Quadro RTX 8000 (48 GB)**, and the efficiency benchmarks were run on an **NVIDIA A100 PCIe 80 GB**.

**Software environment** (versions used by the authors; a Conda env with Python 3.10 is recommended):

| Package | Version |
| --- | --- |
| Python | 3.10 |
| torch | 2.9.1+cu130 |
| torchvision | 0.24.1+cu130 |
| torchaudio | 2.9.1+cu130 |
| numpy | 1.26.4 |
| scipy | 1.15.3 |
| librosa | 0.9.1 |
| soundfile | 0.13.1 |
| h5py | 3.8.0 |
| matplotlib | 3.5.3 |
| transformers | 5.2.0 |
| yamlargparse | 1.31.1 |
| tensorboard | 2.20.0 |
| tqdm | 4.67.3 |
| safe-gpu | 2.0.0 |
| setuptools | 81.0.0 |
| mamba-ssm | 2.3.0 |
| causal-conv1d | 1.6.0 |
| intervaltree | latest (for dscore) |
| tabulate | latest (for dscore) |

> **Note:** No `requirements.txt`, `setup.py`, or `environment.yml` is provided in the repository; dependencies are installed manually as shown in **Installation**. The environment is layered on an **NVIDIA PyTorch (NGC) / CUDA 13.0** base.

**External tools (not vendored):**

- **Kaldi** — <https://github.com/kaldi-asr/kaldi>, used to build the Kaldi-style data directories (see [`diamamba/scripts/prepare_data_dir.sh`](diamamba/scripts/prepare_data_dir.sh)) and for source-speech VAD when generating simulated conversations.
- **dscore** — <https://github.com/nryant/dscore>, used for DER scoring (requires `intervaltree` and `tabulate`). The per-corpus `score.py` files invoke it as `<WORKSPACE>/dscore/score.py`; clone it accordingly or edit the path at the top of each `score.py`.
- **EEND_dataprep** — <https://github.com/BUTSpeechFIT/EEND_dataprep>, used to generate the simulated training conversations (see **Dataset preparation → Simulated training data**).

---

## Installation

The environment below is the one used by the authors, layered on top of an **NVIDIA PyTorch (NGC) / CUDA 13.0** base. The Mamba kernels must be installed with `--no-build-isolation`, and NumPy is force-reinstalled to `1.26.4` after the Mamba packages to avoid ABI conflicts.

### 1. Python environment and dependencies

```bash
# --- Miniconda (skip if conda is already available) ---
mkdir workspace && cd workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda --version

# --- Create and activate the environment ---
conda create -n DiaPer310 python=3.10 -y
conda activate DiaPer310

# --- PyTorch (CUDA 13.0 wheels) ---
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130

# --- Core scientific / audio stack ---
pip install scipy==1.15.3
pip install soundfile==0.13.1
pip install librosa==0.9.1

# --- Training / utility dependencies ---
pip install safe-gpu==2.0.0
pip install tensorboard==2.20.0
pip install yamlargparse==1.31.1
pip install h5py==3.8.0
pip install matplotlib==3.5.3

# --- Build tools and a pinned NumPy (installed before the Mamba kernels) ---
python -m pip install --upgrade pip wheel
python -m pip install setuptools==81.0.0
python -m pip install numpy==1.26.4 --force-reinstall

# --- Mamba selective state-space kernels (CUDA, no build isolation) ---
pip install causal-conv1d==1.6.0 --no-build-isolation
pip install mamba-ssm==2.3.0 --no-build-isolation
```

Verify the install:

```bash
python -c "
import torch
from causal_conv1d import causal_conv1d_fn
from mamba_ssm import Mamba
print('causal-conv1d OK')
print('mamba-ssm OK')
print('PyTorch version:', torch.__version__)
print('PyTorch CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
"
```

### 2. Kaldi (for building data directories)

```bash
cd /root/workspace
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi/tools
extras/check_dependencies.sh          # install any missing OS packages it reports, then rerun
extras/install_openblas.sh
make -j "$(nproc)"
cd ../src
./configure --shared --mathlib=OPENBLAS --openblas-root=<KALDI_ROOT>/tools/OpenBLAS/install
make depend -j "$(nproc)"
make -j "$(nproc)"
```

### 3. dscore (for DER scoring)

```bash
git clone https://github.com/nryant/dscore /workspace/dscore
pip install intervaltree
pip install tabulate
```

### 4. This repository

```bash
git clone https://github.com/codencognition/diamamba.git
```

The training and inference scripts import their internal modules as `backend.*` and `common_utils.*`. Running them as `python diamamba/src/<script>.py …` (or with an absolute path such as `python /workspace/diamamba/src/train.py …`) works because Python automatically adds the script's own directory (`diamamba/src/`) to the import path.

---

## Dataset preparation

DiaMamba uses **Kaldi-style data directories** (`wav.scp`, `segments`, `utt2spk`, `spk2utt`, `rttm`, `reco2dur`), exactly as in DiaPer.

**Datasets used in the paper** (obtain each from its original provider under its own license):

| Dataset | Domain | Speakers | Role |
| --- | --- | --- | --- |
| LibriSpeech (+ MUSAN, RIR) | Simulated conversations | 1–10 | Pretraining (Phases 1–2) |
| CALLHOME (CABank English CallHome) | Telephone | 2–6 | Fine-tuning / evaluation |
| AMI Headset | Meeting | 3–5 | Fine-tuning / evaluation |
| AISHELL-4 | Meeting (Mandarin) | 4–8 | Fine-tuning / evaluation |
| RAMC (MagicData-RAMC) | Telephone (Mandarin) | 2 | Fine-tuning / evaluation |
| VoxConverse | YouTube / in-the-wild | 2–20+ | Evaluation only (no adaptation partition) |

### Simulated training data (Phases 1–2)

The simulated multi-speaker conversations used for pretraining are generated with the **simulated-conversation recipe** of Landini et al. (Interspeech 2022), available at **[BUTSpeechFIT/EEND_dataprep](https://github.com/BUTSpeechFIT/EEND_dataprep)** (Mini LibriSpeech recipe under `v2/MiniLibriSpeech/`). The generation itself is external to this repository; the DiaMamba training scripts consume the resulting Kaldi directories / precomputed features.

**How it works.** The recipe draws source speech from **LibriSpeech** (or Mini LibriSpeech) and mixes it into conversations whose pause and overlap statistics are sampled from a reference **RTTM**. Only the RTTM's *timing table* is used to estimate the distributions of pause lengths, overlap lengths, and their frequencies — **no audio from the reference corpus is read**. Any conversational-telephone-speech RTTM works (e.g. DIHARD 3 dev CTS, CALLHOME). **MUSAN** and **RIR** corpora provide noise/reverberation augmentation.

**What you must provide** before running the recipe:

| Item | Source |
| --- | --- |
| Kaldi (built) | <https://github.com/kaldi-asr/kaldi> — set `KALDIDIR` |
| LibriSpeech / Mini LibriSpeech (`train-clean-5`, `dev-clean-2`) | <https://www.openslr.org/31> — set `LSDIR` to its parent dir |
| One reference RTTM (e.g. DIHARD 3 dev CTS, CALLHOME) | provider of your choice — set `RTTMS_FILE` |
| MUSAN | <https://www.openslr.org/17> — set `NOISES_SCP` to `<MUSAN_DIR>/data/musan_noise_bg/wav.scp` |

**Run the recipe** (paths configured in `v2/MiniLibriSpeech/config_variables.sh`):

```bash
cd v2/MiniLibriSpeech          # inside the EEND_dataprep checkout
pip install -r requirements.txt
./prepareKaldidata_MiniLibriSpeech.sh
./generate_data.sh
```

The paper's recipe scales this to a 2500 h two-speaker set (Phase 1) and a 4000 h 1–10-speaker set (Phase 2). After generation, build features with `process_data.py` (Step 2 below) and point the training configs at them.

**Step 1 — Build a Kaldi data directory** from wav + reference RTTM lists. Edit the paths at the top of the script, then run:

```bash
bash diamamba/scripts/prepare_data_dir.sh
```

If speaker labels in `utt2spk`/`spk2utt` need rebuilding for a corpus, use:

```bash
bash diamamba/scripts/prepare_data_post_processing.sh
```

The `benchmarks/<corpus>/` folders contain the split lists (`train.list`, `dev.list`, `test.list`, etc.) and the `train_dev_test_split.py` / dataset scripts used per corpus.

**Step 2 — Precompute and cache features** so training can use the fast in-memory loader. Edit the paths in the matching config, then run:

```bash
# Uses process_data.py; output layout matches the *_features_dir paths in the training configs
python diamamba/src/process_data.py -c diamamba/configs/process_data_ramc.yaml
```

This writes pickled feature batches to `<features_output_path>/batchsize<macro_batchsize>/train` and `.../dev`, which the training configs reference via `train_features_dir` / `valid_features_dir`. Configs are provided for each corpus: `process_data.yaml`, `process_data_ramc.yaml`, `process_data_aishell4.yaml`, `process_data_amiheadset.yaml`, `process_data_callhome.yaml`.

> An alternative script, [`precompute_features.py`](diamamba/src/precompute_features.py), writes flat `batch_*.pkl` files to a directory given by `--output-features-dir` and takes `--split {train,validation}`. Both formats are readable by `PrecomputedDiarizationDataset`; `process_data.py` is the one whose output layout matches the shipped configs.

If you do **not** precompute features, leave `train_features_dir`/`valid_features_dir` unset and instead set `train_data_dir`/`valid_data_dir` to Kaldi directories — `train.py` will fall back to on-the-fly feature extraction via `KaldiDiarizationDataset`.

---

## Configuration

All stages are driven by YAML files in [`diamamba/configs/`](diamamba/configs/) and parsed with `yamlargparse`; any YAML key can be overridden on the command line (e.g. `--num-frames 3600`). Key shared settings: `feature_dim: 40`, `context_size: 7`, `subsampling: 10`, `input_transform: logmel_meannorm`, `sampling_rate: 16000`, `model_type: AttractorPerceiver`, `n_attractors: 10`, `n_latents: 128`, `mamba_d_state: 16`, `mamba_expand: 4`, `attention_layer_indices: []` (pure BiMamba encoder).

| Config | Stage | Notable settings |
| --- | --- | --- |
| `train_spks_2.yaml` | Phase 1 — 2-speaker pretraining | `num_speakers: 2`, `num_frames: 600`, `optimizer: noam`, `max_epochs: 200` |
| `adapt_morespeakers_spks_10.yaml` | Phase 2 — 1–10-speaker adaptation | `num_speakers: 10`, `num_frames: 600`, `optimizer: noam`, `init_epochs: 191-200` |
| `finetune_adaptedmorespeakers_frames_3600_ramc.yaml` | Phase 3 — RAMC fine-tuning | `optimizer: adam`, `lr: 1e-5`, `num_frames: 3600`, `init_epochs: 91-100` |
| `finetune_adaptedmorespeakers_frames_3600_aishell4.yaml` | Phase 3 — AISHELL-4 | `num_speakers: 8`, `num_frames: 3600` |
| `finetune_adaptedmorespeakers_frames_3600_amiheadset.yaml` | Phase 3 — AMI Headset | `num_speakers: 7`, `num_frames: 3600` |
| `finetune_adaptedmorespeakers_frames_3600_callhome.yaml` | Phase 3 — CALLHOME | `num_speakers: 6`, `num_frames: 3600` |
| `infer_16k_10attractors.yaml` | Inference | `estimate_spk_qty_thr: 0.5`, `threshold: 0.5`, `median_window_length: 1` |
| `process_data*.yaml` | Feature caching | `num_frames: 3600`, `macro_batchsize: 1` |

Before running any stage, edit the absolute paths in the chosen config (`*_features_dir`, `*_data_dir`, `output_path`, `models_path`, `rttms_dir`, `init_model_path`) to match your machine. The shipped configs contain author-specific paths under `/root/workspace` and `/workspace`.

> **Note on sequence length:** the paper's final Phase-3 recipe uses **six-minute** training sequences (`num_frames: 3600`); the fine-tune configs shipped here are the `…_3600` (three-minute) variants. To reproduce the paper's headline numbers, override `--num-frames 3600` (which requires a larger-memory GPU). See **Limitations and known issues**.

---

## Training

DiaMamba is trained with a single entry point, `diamamba/src/train.py`, following a three-phase curriculum in which each phase initializes the next (via `init_model_path` / `init_epochs`). Training resumes automatically from the latest checkpoint if one exists in `output_path/models`. Progress is logged to TensorBoard under `output_path/tensorboard`.

```bash
# Phase 1 — pretrain on 2-speaker simulated conversations
python diamamba/src/train.py -c diamamba/configs/train_spks_2.yaml

# Phase 2 — adapt to 1–10 speakers (initialized from Phase 1)
python diamamba/src/train.py -c diamamba/configs/adapt_morespeakers_spks_10.yaml

# Phase 3 — in-domain fine-tuning on a target corpus (initialized from Phase 2)
python diamamba/src/train.py -c diamamba/configs/finetune_adaptedmorespeakers_frames_3600_ramc.yaml
```

To reproduce the paper's six-minute fine-tuning, add `--num-frames 3600`:

```bash
python diamamba/src/train.py \
    -c diamamba/configs/finetune_adaptedmorespeakers_frames_3600_aishell4.yaml \
    --num-frames 3600
```

Monitor training:

```bash
tensorboard --logdir <output_path>/tensorboard
```

Checkpoints are written to `<output_path>/models/checkpoint_<epoch>.tar`.

---

## Inference

Two inference entry points are provided. Both average the checkpoints in `--models-path` over the epoch range given by `--epochs` (e.g. `41-50`, or comma-separated individual epochs), threshold posteriors at `--threshold` (default 0.5), retain attractors with existence probability `> estimate_spk_qty_thr`, and apply a median filter of width `--median-window-length`. RTTMs are written under a nested, hyperparameter-encoded directory inside `--rttms-dir`.

**Whole-directory inference** (Kaldi data dir):

```bash
python diamamba/src/infer.py \
    -c diamamba/configs/infer_16k_10attractors.yaml \
    --infer-data-dir <kaldi_data_dir> \
    --models-path models/ramc \
    --epochs 1041-1050 \
    --rttms-dir <output_rttms_dir>
```

**Single-file inference** (no Kaldi directory required):

```bash
python diamamba/src/infer_single_file.py \
    -c diamamba/configs/infer_16k_10attractors.yaml \
    --wav-dir <dir_with_wavs> \
    --wav-name <file_id_without_extension> \
    --models-path models/ramc \
    --epochs 1041-1050 \
    --subsampling 10 \
    --median-window-length 11 \
    --rttms-dir <output_rttms_dir>
```

`infer_single_file_gpu_time.py` is the same as `infer_single_file.py` but additionally reports GPU forward-pass timing (used for the efficiency measurements). Set `--gpu 1` to run on GPU; `--plot-output True` (with `--ref-rttms-dir`) writes diagnostic plots. The number of active speakers can be forced with `--estimate-spk-qty N` instead of the threshold.

---

## Evaluation

Evaluation uses **DER** computed with the external [`dscore`](https://github.com/nryant/dscore) tool, matching the DiaPer protocol. Collars follow each corpus convention: **0.25 s** for CALLHOME and VoxConverse; **0 s** for AMI Headset, RAMC, and AISHELL-4.

Each `benchmarks/<corpus>/score.py` orchestrates the full evaluation for that corpus: it runs `infer_single_file.py` on every test recording, concatenates the hypothesis and reference RTTMs, and calls `dscore`. Edit the paths and `MODELS_PATH` / `EPOCHS` / `COLLAR` at the top of the script, then run e.g.:

```bash
python benchmarks/ramc/score.py
```

The DER (and its miss / false-alarm / confusion breakdown) is printed and saved to `benchmarks/<corpus>/scoring_mamba.log`. To score manually:

```bash
python dscore/score.py -r <combined_ref.rttm> -s <combined_hyp.rttm> --collar 0
```

The repository already ships the hypothesis RTTMs and `scoring_mamba.log` files for five fine-tuning seeds (`FT_Seed_3`, `FT_Seed_10`, `FT_Seed_15`, `FT_Seed_20`, `FT_Seed_25`) and a no-fine-tuning (`noFT`) run per corpus.

---

## Reproducing the paper's results

The three-phase protocol, with the data pipeline, feature extraction, and scoring held identical to DiaPer:

1. **Prepare data** — build Kaldi dirs (`prepare_data_dir.sh`) and precompute features (`process_data.py`) for the simulated sets and each target corpus.
2. **Phase 1** — pretrain on the 2500 h two-speaker simulated set: `train.py -c train_spks_2.yaml` (`num_frames: 600`, Noam, batch size 32, up to 200 epochs).
3. **Phase 2** — adapt to 1–10 speakers on the 4000 h simulated set: `train.py -c adapt_morespeakers_spks_10.yaml`, initialized from Phase 1 epochs 191–200.
4. **Phase 3** — fine-tune per corpus with Adam (`lr 1e-4` in the paper; `1e-5` in the RAMC config) at **`num_frames: 3600`** (override the shipped `3600` configs), initialized from Phase 2 epochs 91–100. Run **five random seeds** per corpus (`3, 10, 15, 20, 25`).
5. **Checkpoint averaging** — average the parameters of **10 consecutive** epoch checkpoints for each evaluation checkpoint (this is what `--epochs a-b` does).
6. **Evaluate** — run the per-corpus `score.py` and report the **median DER across the five seeds** (as in the paper).

> Absolute efficiency numbers (FLOPs / memory / latency) were measured on an A100 80 GB with `torch.profiler` and CUDA-event timing; hardware differences will shift absolute figures, though the asymptotic advantage is structural.

---

## Main results (reported in the paper)

> All figures below are **reported in the paper**. DiaPer figures are reproduced by the authors from Landini et al. (2024). DiaMamba figures are the **median over five seeds**. Δ = DiaMamba − DiaPer (negative is better).

### DER (%) after in-domain fine-tuning

| Corpus | Collar | VAD+VBx+OSD | DiaPer | **DiaMamba** | Δ |
| --- | --- | --- | --- | --- | --- |
| CALLHOME (CABank re-release) | 0.25 | – | – | **7.57** | – |
| AMI Headset | 0 | 22.40 | 32.90 | **28.53** | −4.37 |
| RAMC | 0 | 18.20 | 21.10 | **17.07** | −4.03 |
| AISHELL-4 | 0 | 15.80 | 41.40 | **32.40** | −9.00 |

### DER (%) without fine-tuning (Phase-2 checkpoint evaluated directly)

| Corpus | Collar | DiaPer | **DiaMamba** | Δ |
| --- | --- | --- | --- | --- |
| CALLHOME | 0.25 | – | **12.87** | – |
| AMI Headset | 0 | 36.40 | **45.98** | +9.58 |
| VoxConverse | 0.25 | 23.20 | **29.86** | +6.66 |
| RAMC | 0 | 38.10 | **33.01** | −5.09 |
| AISHELL-4 | 0 | 48.20 | **45.76** | −2.44 |

### Statistical significance of the fine-tuned gains

| Corpus (FT) | DiaMamba (mean ± std, %) | DiaPer (%) | ΔDER | 95% CI | p | Cohen's d |
| --- | --- | --- | --- | --- | --- | --- |
| AMI Headset | 29.20 ± 1.33 | 32.90 | −3.70 | [−5.35, −2.05] | 3.4×10⁻³ | −2.78 |
| RAMC | 17.22 ± 0.59 | 21.10 | −3.88 | [−4.61, −3.15] | 1.2×10⁻⁴ | −6.58 |
| AISHELL-4 | 32.70 ± 2.12 | 41.40 | −8.70 | [−11.33, −6.07] | 7.8×10⁻⁴ | −4.10 |

### Parameter count and frame-encoder complexity

| Model | Params (M) | Time | Memory |
| --- | --- | --- | --- |
| DiaPer | 4.34 | O(T²D) | O(T²) |
| **DiaMamba** | 4.35 | **O(TD)** | **O(T)** |

**Efficiency highlights (reported):** at the longest tested recording (49 min), DiaMamba uses ≈**86.5 % fewer FLOPs** (277 vs 2,051 GFLOPs), up to ≈**116× lower peak GPU memory** (0.60 GB vs 69.4 GB), and stays below 1 GB of VRAM across all tested lengths; GPU inference latency is near-constant (~0.23 s), crossing DiaPer near 30 min (≈1.7× faster on long recordings).

### Encoder-design ablation (Phase-1, 2-speaker simulated dev set)

| Configuration | Best DER (%) |
| --- | --- |
| **M4+A0** (pure BiMamba, `d_state=16`, expand=4, dropout=0.15) — proposed | **1.52** |
| M4+A0, `d_state=16`, expand=2 | 1.65 |
| M4+A0, `d_state=32`, expand=4 | 1.77 |
| M3+A1 (hybrid), best | 1.92 |
| M0+A4 (self-attention baseline, DiaPer encoder) | 1.94 |

| Component removed from the M4+A0 winner | Best DER (%) | ΔDER |
| --- | --- | --- |
| Full model (proposed) | 1.52 | — |
| − bidirectionality (forward-only Mamba) | 2.30 | +0.78 |
| − frame–attractor conditioning | 4.20 | +2.68 |
| − ±7-frame temporal context | 3.10 | +1.58 |

---

## Pretrained checkpoints

Pretrained checkpoints ship in [`models/`](models/). Each folder holds **10 consecutive epoch checkpoints** intended to be averaged at inference via `--epochs <a-b>`.

| Folder | Stage | `--epochs` |
| --- | --- | --- |
| `models/simulatedConversationLibriSpeechTrain2Spks/` | Phase 1 (2-speaker pretraining) | `191-200` |
| `models/simulatedConversationLibriSpeechAdapt10Spks/` | Phase 2 (1–10-speaker adaptation) | `91-100` |
| `models/amiheadset/` | Phase 3 fine-tuned (AMI Headset) | `141-150` |
| `models/aishell4/` | Phase 3 fine-tuned (AISHELL-4) | `341-350` |
| `models/ramc/` | Phase 3 fine-tuned (RAMC) | `1041-1050` |
| `models/cabanksenglishcallhome/` | Phase 3 fine-tuned (CALLHOME) | `641-650` |

Example (RAMC):

```bash
python diamamba/src/infer_single_file.py \
    -c diamamba/configs/infer_16k_10attractors.yaml \
    --wav-dir <dir_with_wavs> --wav-name <file_id> \
    --models-path models/ramc --epochs 1041-1050 \
    --median-window-length 11 --rttms-dir <output_rttms_dir>
```

> The shipped checkpoints correspond to the `…_3600` fine-tuning configs.

---

## Limitations and known issues

- **No-fine-tuning gap (paper limitation).** Without in-domain adaptation, DiaMamba trails DiaPer on the wide-band Western corpora (up to +9.58 DER on AMI Headset), though it already leads on both Mandarin corpora. The advantage over DiaPer is realized uniformly only *after* fine-tuning, so DiaMamba is more sensitive to adaptation-data availability than DiaPer.
- **Constrained pretraining budget.** Phase 2 used ~4,000 h of simulated data versus ~22,500 h for the original DiaPer; a larger Phase-2 corpus is the natural route to narrowing the no-fine-tuning gap.
- **Sequence-length mismatch between paper and shipped configs.** The paper's headline recipe uses `num_frames: 3600` (six minutes), but the fine-tune configs in this repo are the `num_frames: 3600` variants and one scoring script (`benchmarks/ramc/score.py`) references a `ft_ramc_f3600` model that is not included. Override `--num-frames 3600` (on sufficient-memory hardware) to match the paper.
- **CALLHOME comparison caveat.** CALLHOME is evaluated on the CABank re-release (not the LDC distribution used by DiaPer), so it is reported as an internal no-FT vs FT comparison rather than a strict like-for-like DiaPer result.
- **Hybrid configs under-trained.** The ablation's M3+A1 hybrid runs were stopped near 100 epochs (vs 200 for the pure-BiMamba family), so their ceiling is not fully established.
- **Inherited hyperparameters.** Encoder depth, number of Perceiver blocks, and latent dimensionality were inherited from DiaPer without re-optimization.
- **GPU-only.** The `mamba-ssm` / `causal-conv1d` kernels require a CUDA GPU; there is no practical CPU path.
- **Manual configuration.** Configs contain author-specific absolute paths and must be edited before use; there is no packaged dependency manifest.
- **External dependencies.** Kaldi (data-dir construction) and `dscore` (scoring) must be installed separately; simulated-conversation generation is not included in this repository.

---

## License

This project is released under the **MIT License**. See [`LICENSE`](LICENSE). The codebase builds on DiaPer and retains the original Hitachi, Ltd. and Brno University of Technology copyright notices, which are also MIT-licensed.

---

## Acknowledgements

- Built directly on **[DiaPer](https://github.com/BUTSpeechFIT/DiaPer)** (F. Landini, M. Diez, T. Stafylakis, L. Burget, *IEEE/ACM TASLP*, 2024), from which the Perceiver-based attractor decoder, frame–attractor conditioning, data pipeline, and training/scoring protocol are inherited. Original code © Hitachi, Ltd. (Yusuke Fujita) and Brno University of Technology (Federico Landini).
- **[Mamba](https://github.com/state-spaces/mamba)** selective state-space models (A. Gu, T. Dao) via the `mamba-ssm` and `causal-conv1d` kernels.
- **Simulated conversations** generated with the [BUTSpeechFIT/EEND_dataprep](https://github.com/BUTSpeechFIT/EEND_dataprep) recipe (Landini et al., Interspeech 2022); **LibriSpeech / Mini LibriSpeech**, **MUSAN**, and **RIR** corpora for pretraining data and augmentation.
- Evaluation datasets: **CALLHOME (CABank English CallHome)**, **AMI Headset**, **AISHELL-4**, **MagicData-RAMC**, and **VoxConverse**.
- Scoring with **[dscore](https://github.com/nryant/dscore)** (N. Ryant).

---

## Citation

If you use DiaMamba, please cite the paper:

```bibtex
@article{rauf_diamamba,
  title   = {DiaMamba: End-to-End Neural Diarization with Bidirectional Mamba
             Encoder and Perceiver Attractor Decoder},
  author  = {Rauf, Usman and Hussain, Basharat},
  year    = {2026},
  note    = {[TODO: add journal/conference, volume, pages, DOI, or arXiv ID]}
}
```

Please also consider citing the DiaPer baseline this work builds on:

```bibtex
@article{landini2024diaper,
  title   = {DiaPer: End-to-End Neural Diarization With Perceiver-Based Attractors},
  author  = {Landini, Federico and Diez, Mireia and Stafylakis, Themos and Burget, Luk\'{a}\v{s}},
  journal = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume  = {32},
  pages   = {3450--3465},
  year    = {2024}
}
```

---

## Contact

- Usman Rauf — usmanraufraja@gmail.com
- National University of Computer and Emerging Sciences (FAST-NUCES)
