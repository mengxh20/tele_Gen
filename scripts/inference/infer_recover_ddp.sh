#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Inference uses torchrun multi-process sample sharding:
# each rank handles a subset of input latent files instead of splitting one sample across GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export MAX_SAMPLES="${MAX_SAMPLES:-4}" # 推理4个样本看看就够了
export RECOVER_RUN_BASE_DIR="${RECOVER_RUN_BASE_DIR:-${PROJECT_ROOT}/reconstruct/experiments/recover_s10_t3_w7_20260415_211426}"
export RECOVER_SECTION="${RECOVER_SECTION:-}"
export RECOVER_ANCHOR="${RECOVER_ANCHOR:-}"
export RECOVER_STEPS="${RECOVER_STEPS:-}"

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

INFER_ARGS+=("$@")

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${INFER_ARGS[@]}"


# 推理完之后自动进行解码恢复视频
python "${PROJECT_ROOT}/reconstruct/decoder.py" -R "${SAVEDIR}/recover_outputs"
