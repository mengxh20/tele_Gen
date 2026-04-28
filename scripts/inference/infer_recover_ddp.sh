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
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
export RECOVER_OVERLAP_LATENTS="${RECOVER_OVERLAP_LATENTS:-2}"
export VAE_DECODE_MODE="${VAE_DECODE_MODE:-auto}"


SAVEDIR="${SAVEDIR:-reconstruct/gen2recon_runs_HE2E_Apr28}"
INFER_ARGS=(
    reconstruct/recover.py infer
    --checkpoint_dir "${SAVEDIR}/checkpoints"
    --output_dir "${SAVEDIR}/recover_outputs"
    --num_inference_steps "${NUM_INFERENCE_STEPS}"
    --recover_overlap_latents "${RECOVER_OVERLAP_LATENTS}"
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


# 推理已经产出 recover_latents，后处理阶段直接解码这份恢复结果用于可视化验收。
python reconstruct/decoder.py \
    -R "${SAVEDIR}/recover_outputs" \
    --input_dir "${SAVEDIR}/recover_outputs/recover_latents" \
    --vae_decode_mode "${VAE_DECODE_MODE}"
