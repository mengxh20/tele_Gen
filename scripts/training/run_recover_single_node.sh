#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export MASTER_ADDR=127.0.0.1
export MASTER_PORT="${MASTER_PORT:-29641}"
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth3}"

# Keep the single-node run close to the stable NCCL test settings.
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SHM_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export HELIOS_BASE_MODEL_PATH=/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base
export RECOVER_INPUT_PATH=/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/codes/tele_Gen/train_dataset2
export RECOVER_OUTPUT_DIR=/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_from_ep120_single_node
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-1000}"

echo "[INFO] role=single-node addr=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} rank=${NODE_RANK}"
echo "[INFO] nproc_per_node=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nccl_ifname=${NCCL_SOCKET_IFNAME}"
echo "[INFO] ib_disable=${NCCL_IB_DISABLE} socket_family=${NCCL_SOCKET_FAMILY} shm_disable=${NCCL_SHM_DISABLE}"
echo "[INFO] checkpoint_every_steps=${CHECKPOINT_EVERY_STEPS}"

bash scripts/training/train_multinode.sh \
  --init_checkpoint_dir /data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_section7_anchor1_steps24/checkpoints/epoch_0120_step_00002520 \
  --section 7 \
  --anchor 1 \
  --steps 24 \
  --history_sizes 5 3 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --epochs 1000 \
  --max_steps 50000 \
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}" \
  --num_workers 0 \
  --lazy_sequence_cache_size 1 \
  --learning_rate 2e-6 \
  --weight_dtype bf16 \
  --fix_anchor_during_denoise \
  --soft_tail_hint_strength 0.35 \
  --soft_tail_hint_step_fraction 0.5 \
  --max_predict_only_gap_sections 2 \
  --single_refresh_gain_threshold 0.12 \
  --dual_refresh_gain_threshold 0.18 \
  --boundary_jump_threshold 0.18 \
  --cut_detection_threshold 0.35 \
  --train_mixed_dual_tail_ratio 0.0 \
  --global_keyframe_codec_type int8_refresh_like \
  --global_keyframe_spatial_factor 1 \
  --dual_head_codec_type soft_pool \
  --dual_head_soft_pool_spatial_factor 5 \
  --dual_tail_codec_type p_delta \
  --dual_tail_p_delta_spatial_factor 4
