#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29647}"
export NNODES=1
export NODE_RANK=0
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth3}"

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

export HELIOS_BASE_MODEL_PATH="${HELIOS_BASE_MODEL_PATH:-/ddn/team/shared/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base}"
export RECOVER_INPUT_PATH="${RECOVER_INPUT_PATH:-/ddn/team/shared/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/codes/tele_Gen/train_dataset2}"
export RECOVER_OUTPUT_DIR="${RECOVER_OUTPUT_DIR:-/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_safeaux10k_from_ep120}"

export MAX_STEPS="${MAX_STEPS:-10000}"
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-1000}"

echo "[INFO] role=single-node-safeaux-10k addr=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} rank=${NODE_RANK}"
echo "[INFO] nproc_per_node=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nccl_ifname=${NCCL_SOCKET_IFNAME}"
echo "[INFO] input=${RECOVER_INPUT_PATH}"
echo "[INFO] output=${RECOVER_OUTPUT_DIR}"
echo "[INFO] max_steps=${MAX_STEPS} checkpoint_every_steps=${CHECKPOINT_EVERY_STEPS}"

bash scripts/training/train_multinode.sh \
  --init_checkpoint_dir /data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_section7_anchor1_steps24/checkpoints/epoch_0120_step_00002520 \
  --section 7 \
  --anchor 1 \
  --steps 24 \
  --history_sizes 5 3 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --epochs 1000 \
  --max_steps "${MAX_STEPS}" \
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}" \
  --num_workers 0 \
  --lazy_sequence_cache_size 1 \
  --learning_rate 5e-7 \
  --max_grad_norm 0.5 \
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
  --dual_tail_p_delta_spatial_factor 4 \
  --flow_loss_weight 1.0 \
  --x_loss_weight 0.005 \
  --noise_loss_weight 0.0 \
  --temporal_delta_loss_weight 0.001 \
  --aux_loss_warmup_steps 2000 \
  --aux_loss_ramp_steps 8000 \
  --aux_loss_clip_value 5.0 \
  --aux_loss_sigma_min 0.05 \
  --aux_loss_sigma_max 0.95
