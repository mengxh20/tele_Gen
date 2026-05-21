#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export MASTER_PORT="${MASTER_PORT:-29668}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"

export HELIOS_BASE_MODEL_PATH="${HELIOS_BASE_MODEL_PATH:-/data_old/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base}"
export INPUT_LATENTS="${INPUT_LATENTS:-/data/heli/code/tele_Gen/reconstruct/latents_nochunk_149}"
export SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-/data/heli/code/tele_Gen/reconstruct/videos}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue10k_from_step50000_section7_anchor1_steps24/checkpoints/epoch_0001_step_00010000}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/heli/code/tele_Gen/reconstruct/experiments2/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_base_safeaux_continue10k_from_step50000_section7_anchor1_steps24/recover_outputs_videoSRC01_04_nochunk149_step10000_cumulative60000_x0int8sf1_headsoft5_tailp4}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"

echo "[INFO] checkpoint=${CHECKPOINT_DIR}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] input_latents=${INPUT_LATENTS}"

torchrun --nproc_per_node="${NPROC_PER_NODE}" --master_port="${MASTER_PORT}" reconstruct/recover.py infer \
  --input_path "${INPUT_LATENTS}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model_path "${HELIOS_BASE_MODEL_PATH}" \
  --weight_dtype bf16 \
  --steps 24 \
  --predict_only_steps 2 \
  --single_refresh_steps 8 \
  --dual_refresh_steps 12 \
  --global_keyframe_codec_type int8_refresh_like \
  --global_keyframe_spatial_factor 1 \
  --dual_head_codec_type soft_pool \
  --dual_head_soft_pool_spatial_factor 5 \
  --dual_tail_codec_type p_delta \
  --dual_tail_p_delta_spatial_factor 4 \
  --soft_tail_hint_strength 0.35 \
  --soft_tail_hint_step_fraction 0.5 \
  --max_samples "${MAX_SAMPLES}"

python reconstruct/decoder.py \
  -R "${OUTPUT_DIR}" \
  --source_video_dir "${SOURCE_VIDEO_DIR}" \
  --base_model_path "${HELIOS_BASE_MODEL_PATH}" \
  --compare_mode both \
  --compute_lpips \
  --lpips_batch_size 8
