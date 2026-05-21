#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export RECOVER_FULL_INPUT_PATH="${RECOVER_FULL_INPUT_PATH:-/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/codes/tele_Gen/train_dataset2}"
export LAZY_CHECK_LOAD_FILES="${LAZY_CHECK_LOAD_FILES:-3}"
export LAZY_CHECK_CACHE_SIZE="${LAZY_CHECK_CACHE_SIZE:-1}"

echo "[INFO] input=${RECOVER_FULL_INPUT_PATH}"
echo "[INFO] load_files=${LAZY_CHECK_LOAD_FILES} cache_size=${LAZY_CHECK_CACHE_SIZE}"

python - <<'PY'
import os
import resource
import time
from pathlib import Path

from reconstruct.recover import (
    LazyPreparedSequenceStore,
    build_codec_config,
    build_sequence_index,
    discover_input_latent_paths,
)


class Args:
    temporal_factor = 2
    spatial_factor = 4
    quant_dtype = "int8"
    keyframe_dtype = "float16"
    global_keyframe_codec_type = "raw"
    global_keyframe_quant_dtype = "int8"
    global_keyframe_spatial_factor = 1
    section_span_latents = 7
    anchor_span_latents = 1
    tail_span_latents = None
    anchor_quant_dtype = "int8"
    anchor_spatial_factor = 1
    dual_head_codec_type = "soft_pool"
    dual_head_anchor_spatial_factor = None
    dual_head_p_delta_spatial_factor = 4
    dual_head_soft_pool_spatial_factor = 5
    adaptive_dual_head_full_ratio = 0.0
    adaptive_dual_head_soft_pool_hard_factor = None
    adaptive_dual_head_soft_pool_hard_ratio = 0.0
    dual_tail_anchor_spatial_factor = None
    dual_tail_codec_type = "p_delta"
    dual_tail_p_delta_spatial_factor = 4
    tail_codec_type = "trilinear"
    learned_codec_hidden_channels = None
    max_predict_only_gap_sections = 2
    single_refresh_gain_threshold = 0.12
    dual_refresh_gain_threshold = 0.18
    boundary_jump_threshold = 0.18
    cut_detection_threshold = 0.35
    target_low_bpp = None


def rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


input_path = Path(os.environ["RECOVER_FULL_INPUT_PATH"])
load_files = int(os.environ.get("LAZY_CHECK_LOAD_FILES", "3"))
cache_size = int(os.environ.get("LAZY_CHECK_CACHE_SIZE", "1"))

start = time.time()
input_root, latent_paths = discover_input_latent_paths(input_path)
print(f"[INFO] discovered files={len(latent_paths)} root={input_root} rss_peak={rss_gb():.3f}GB", flush=True)

index_start = time.time()
entries = build_sequence_index(latent_paths)
windows = sum(max(0, (entry.total_latent_frames - 2) // 7 + 1) for entry in entries)
print(
    f"[INFO] built lightweight index entries={len(entries)} approx_windows={windows} "
    f"seconds={time.time() - index_start:.2f} rss_peak={rss_gb():.3f}GB",
    flush=True,
)

store = LazyPreparedSequenceStore(
    entries=entries,
    codec_config=build_codec_config(Args()),
    cache_size=cache_size,
)
for seq_idx in range(min(load_files, len(entries))):
    load_start = time.time()
    sequence = store[seq_idx]
    print(
        f"[INFO] lazy-loaded seq={seq_idx} frames={sequence.clean_full_latents.shape[1]} "
        f"shape={tuple(sequence.clean_full_latents.shape)} "
        f"seconds={time.time() - load_start:.2f} rss_peak={rss_gb():.3f}GB",
        flush=True,
    )

print(f"[OK] lazy full-data check completed total_seconds={time.time() - start:.2f}", flush=True)
PY
