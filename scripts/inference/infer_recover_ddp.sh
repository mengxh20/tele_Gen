#!/bin/bash
set -euo pipefail

# Inference uses torchrun multi-process sample sharding:
# each rank handles a subset of input latent files instead of splitting one sample across GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export MAX_SAMPLES="${MAX_SAMPLES:-4}" # 推理4个样本看看就够了
export INPUT_PATH="${INPUT_PATH:-reconstruct/latents}"


SAVEDIR="reconstruct/gen2recon_runs_E2E"
INFER_ARGS=(
    reconstruct/recover.py infer
    --input_path "${INPUT_PATH}"
    --checkpoint_dir "${SAVEDIR}/checkpoints"
    --output_dir "${SAVEDIR}/recover_outputs"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
    INFER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

INFER_ARGS+=("$@")

torchrun \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${INFER_ARGS[@]}"


# # 推理完之后自动进行解码恢复视频
# python reconstruct/decoder.py -R ${SAVEDIR}/recover_outputs

# 推理完之后自动走接收端链路：
# low_latents -> recover network -> recover_latents -> video
python reconstruct/real_decoder.py \
    -R "${SAVEDIR}/recover_outputs" \
    --checkpoint_dir "${SAVEDIR}/checkpoints"
