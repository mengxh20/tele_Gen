#!/usr/bin/env bash
set -euo pipefail

cd /ddn/team/shared/heli/code/tele_Gen

export PATH=/ddn/team/shared/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export MASTER_ADDR=10.11.1.109
export MASTER_PORT=29631
export NNODES=2
export NODE_RANK=1
export NPROC_PER_NODE=8
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_ADDR_FAMILY="${NCCL_IB_ADDR_FAMILY:-AF_INET}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-1}"
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET}"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1

export HELIOS_BASE_MODEL_PATH=/ddn/team/shared/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base
export RECOVER_INPUT_PATH=/ddn/team/shared/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/codes/tele_Gen/train_dataset2
export RECOVER_OUTPUT_DIR=/ddn/team/shared/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_flowonly_bs2_from_ep120
export CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-500}"

echo "[INFO] role=worker addr=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} rank=${NODE_RANK}"
echo "[INFO] nproc_per_node=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nccl_ifname=${NCCL_SOCKET_IFNAME:-auto}"
echo "[INFO] ib_disable=${NCCL_IB_DISABLE} ib_hca=${NCCL_IB_HCA:-auto} ib_gid_index=${NCCL_IB_GID_INDEX:-auto} socket_family=${NCCL_SOCKET_FAMILY} shm_disable=${NCCL_SHM_DISABLE} debug_subsys=${NCCL_DEBUG_SUBSYS}"
echo "[INFO] checkpoint_every_steps=${CHECKPOINT_EVERY_STEPS}"

bash scripts/training/train_multinode.sh \
  --init_checkpoint_dir /ddn/team/shared/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_headsoft_tailp_section7_anchor1_steps24/checkpoints/epoch_0120_step_00002520 \
  --section 7 \
  --anchor 1 \
  --steps 24 \
  --history_sizes 5 3 1 \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --epochs 1000 \
  --max_steps 50000 \
  --checkpoint_every_steps "${CHECKPOINT_EVERY_STEPS}" \
  --num_workers 0 \
  --lazy_sequence_cache_size 1 \
  --learning_rate 1e-6 \
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
  --x_loss_weight 0.0 \
  --noise_loss_weight 0.0 \
  --temporal_delta_loss_weight 0.0
