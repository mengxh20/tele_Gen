#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Inference uses torchrun multi-process sample sharding:
# each rank handles a subset of latent files.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"

# Limit sample count for quick validation by default; unset or override for full set.
export MAX_SAMPLES="${MAX_SAMPLES:-4}"
export RECOVER_RUN_BASE_DIR="${RECOVER_RUN_BASE_DIR:-${PROJECT_ROOT}/reconstruct/gen2recon_runs_CNN_3_section4_anchor1_steps24}"
export RECOVER_SECTION="${RECOVER_SECTION:-}"
export RECOVER_ANCHOR="${RECOVER_ANCHOR:-}"
export RECOVER_STEPS="${RECOVER_STEPS:-}"

# Shared output root for infer + decode.
SAVEDIR="${RECOVER_RUN_BASE_DIR}"
if [[ -n "${RECOVER_SECTION}" && -n "${RECOVER_ANCHOR}" && -n "${RECOVER_STEPS}" ]]; then
    SAVEDIR="${RECOVER_RUN_BASE_DIR}_section${RECOVER_SECTION}_anchor${RECOVER_ANCHOR}_steps${RECOVER_STEPS}"
fi

INFER_ARGS=(
    "${PROJECT_ROOT}/reconstruct/recover.py" infer
    --checkpoint_dir "${SAVEDIR}/checkpoints"
    --output_dir "${SAVEDIR}/recover_outputs"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    INFER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

if [[ -n "${RECOVER_STEPS}" ]]; then
    INFER_ARGS+=(--steps "${RECOVER_STEPS}")
fi

# Forward extra CLI args to recover infer.
INFER_ARGS+=("$@")

echo "[pipeline] Step 1/2: recover infer"
torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${INFER_ARGS[@]}"

# Decoder controls:
#   RUN_DECODER=0 to skip decode
#   DECODER_COMPARE_MODE=off|full|sections|both
#   SOURCE_VIDEO_DIR=/path/to/source/videos (for compare export)
#   DECODER_COMPARE_OUTPUT_DIR=/path/to/output/duibi_videos (optional)
#   DECODER_DEVICE=cuda|cpu
RUN_DECODER="${RUN_DECODER:-1}"
DECODER_COMPARE_MODE="${DECODER_COMPARE_MODE:-both}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-${PROJECT_ROOT}/reconstruct/videos}"
DECODER_DEVICE="${DECODER_DEVICE:-cuda}"
DECODER_COMPARE_OUTPUT_DIR="${DECODER_COMPARE_OUTPUT_DIR:-}"

if [[ "${RUN_DECODER}" == "1" ]]; then
    echo "[pipeline] Step 2/2: decode recover latents to videos"
    DECODER_ARGS=(
        "${PROJECT_ROOT}/reconstruct/decoder.py"
        -R "${SAVEDIR}/recover_outputs"
        --device "${DECODER_DEVICE}"
        --compare_mode "${DECODER_COMPARE_MODE}"
        --source_video_dir "${SOURCE_VIDEO_DIR}"
    )

    if [[ -n "${DECODER_COMPARE_OUTPUT_DIR}" ]]; then
        DECODER_ARGS+=(--compare_output_dir "${DECODER_COMPARE_OUTPUT_DIR}")
    fi

    python "${DECODER_ARGS[@]}"
else
    echo "[pipeline] Decoder skipped because RUN_DECODER=${RUN_DECODER}"
fi
