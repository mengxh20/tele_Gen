# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TeleGen is a **generative video compression and recovery system** built on top of the Helios video generation model (14B params). The primary goal is low-bitrate video compression, transmission, and high-quality recovery — not general-purpose video generation (t2v/i2v/v2v).

The pipeline operates on Helios VAE latents: encode video → compress to low bitrate → recover via generative model → decode back to video.

## Architecture

Two-layer structure:

- **`helios/`** — Core model library (transformer, Triton kernels, pipelines, scheduler, dataset utils). Originally the upstream Helios capability base, but can be modified when needed to unlock model capabilities for compression and recovery tasks.
- **`reconstruct/`** — Primary development target. All compression, recovery, latent codec, and related logic belongs here.

### Core Pipeline (`reconstruct/`)

1. **`encoder.py`** — Video → Helios VAE latents (`.pt`)
2. **`recover.py`** — Main entry for both training and inference. Compresses clean latents into low-bitrate representation (spatial/temporal downsampling + quantization + learned CNN codec), then recovers quality using Helios stage1 transformer + scheduler
3. **`decoder.py`** — Latents → video (for visualization/validation only, not the main pipeline)
4. **`real_decoder.py`** — Receiver-side decode: reads `low_latents`, runs recover network, outputs video (simulates real transmission chain)
5. **`latent_io.py`** — Latent I/O, format validation, chunking utilities
6. **`codec_gop.py`** — GOP-level codec: anchor/tail encoding, quantization, motion scoring
7. **`learned_codec.py`** — Learned CNN-based tail codec (compressai-based with hyperprior Gaussian entropy model)

### Key Helios Components Used

- Stage1 conditional assembly, transformer (`helios/modules/transformer_helios.py`), scheduler (`helios/scheduler/scheduling_helios.py`)
- Custom Triton kernels in `helios/modules/helios_kernels/` (attention dispatch, fused RMSNorm, RoPE, tiled linear)
- Empty prompt encoding for unconditional generation

## Commands

### Installation

```bash
conda create -n helios python=3.11.2 && conda activate helios
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
bash install.sh
```

### Training (Compression-Recovery — primary)

```bash
bash scripts/training/train_recover_ddp.sh
```

Runs `reconstruct/recover.py train` via Accelerate + DeepSpeed ZeRO-3. Output goes to `reconstruct/gen2recon_runs_*`. Produces `transformer_full.pt`, `recover_config.json`, `train_metrics.json`, and `checkpoints/latest`.

### Training (Upstream Helios — secondary)

```bash
bash scripts/training/train_ddp.sh          # DDP
bash scripts/training/train_deepspeed.sh     # DeepSpeed ZeRO-2
```

Three-stage progressive pipeline with configs in `scripts/training/configs/`.

### Inference (Compression-Recovery)

```bash
bash scripts/inference/infer_recover_ddp.sh
```

Runs `reconstruct/recover.py infer` via torchrun, then decodes via `reconstruct/decoder.py`. Outputs: `low_latents/`, `recover_latents/`, `metrics/`.

### Inference (Upstream Helios)

```bash
bash scripts/inference/helios-base_t2v.sh    # text-to-video
bash scripts/inference/helios-base_i2v.sh    # image-to-video
bash scripts/inference/helios-base_v2v.sh    # video-to-video
```

### Evaluation

```bash
cd eval && bash run_metrics.sh       # single GPU
cd eval && bash run_metrics_ddp.sh   # multi-GPU
```

### Quick Start (train + infer recovery)

```bash
bash run.sh
```

## Key Patterns

- **LoRA**: rank 128, alpha 128, applied to all linear layers except `down`/`up`
- **Mixed precision**: bf16 throughout
- **Gradient checkpointing**: enabled for memory efficiency
- **Section/window organization**: videos split into sections with anchors and tails; recovery operates section-by-section with overlapping history
- **Codec modes**: `raw`, `quantized_int8`, trilinear tail codec, learned CNN-based tail codec
- **Dynamic rate control**: motion-score-based quality allocation (`latent_delta_l1_norm`) with configurable tail quality bounds
- **Latent format versioning**: multiple versions tracked in `latent_io.py` (`helios_vae_latent_v1/v2`, `helios_low_latent_v1/v2/v3/v4`)

## Conventions

- `reconstruct/` remains the primary development target; new code defaults there
- `helios/` can be modified when it helps unlock model capabilities — no special justification required
- Prefer minimal necessary changes; avoid unrelated refactoring
- Training, inference, and evaluation changes should be framed in terms of: bitrate efficiency, recovery quality, temporal consistency, training stability
- Artifacts: `entropy_metrics/` retains entropy coding bitrate stats; `enc_latents/` only saves `.bin` when `--save_entropy_bin` is explicitly set
- For simulating real receiver-side decoding without writing `.bin` files, use `real_decoder.py` which reads `low_latents` and runs recovery in-memory
