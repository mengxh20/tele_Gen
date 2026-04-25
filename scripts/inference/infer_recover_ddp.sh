#!/bin/bash
set -euo pipefail

# Inference uses torchrun multi-process sample sharding:
# each rank handles a subset of input latent files instead of splitting one sample across GPUs.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29531}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export MAX_SAMPLES="${MAX_SAMPLES:-4}" # 推理4个样本看看就够了


SAVEDIR="reconstruct/gen2recon_runs_HE2E_dyn"
INFER_ARGS=(
    reconstruct/recover.py infer
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
# 直接以 low_latents 作为输入，在内存中模拟真实运输码流，再 recover -> video。
# 这样不会依赖 enc_latents/.bin，也能避免旧 .bin 残留影响当前结果。
python reconstruct/real_decoder.py \
    -R "${SAVEDIR}/recover_outputs" \
    --input_dir "${SAVEDIR}/recover_outputs/low_latents" \
    --transport_input_mode simulate_entropy \
    --checkpoint_dir "${SAVEDIR}/checkpoints"
