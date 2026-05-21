#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29651}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_from_ep120_section7_anchor1_steps24/checkpoints/epoch_0001_step_00002000}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_from_ep120_section7_anchor1_steps24/recover_outputs_step2000_x0int8sf1_headsoft5_tailp4}"

# Keep default input aligned with the 0506-2 comparison run. Override this env var
# when you want to infer on the 1w no-chunk dataset itself.
RECOVER_INPUT_PATH="${RECOVER_INPUT_PATH:-${PROJECT_ROOT}/reconstruct/latents}"
HELIOS_BASE_MODEL_PATH="${HELIOS_BASE_MODEL_PATH:-/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base}"
MAX_SAMPLES="${MAX_SAMPLES-4}"

echo "[INFO] checkpoint_dir=${CHECKPOINT_DIR}"
echo "[INFO] output_dir=${OUTPUT_DIR}"
echo "[INFO] input_path=${RECOVER_INPUT_PATH}"
echo "[INFO] cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nproc_per_node=${NPROC_PER_NODE}"

INFER_ARGS=(
  "${PROJECT_ROOT}/reconstruct/recover.py" infer
  --input_path "${RECOVER_INPUT_PATH}"
  --checkpoint_dir "${CHECKPOINT_DIR}"
  --output_dir "${OUTPUT_DIR}"
  --base_model_path "${HELIOS_BASE_MODEL_PATH}"
  --steps 24
  --predict_only_steps 2
  --single_refresh_steps 8
  --dual_refresh_steps 12
  --global_keyframe_codec_type int8_refresh_like
  --global_keyframe_spatial_factor 1
  --dual_head_codec_type soft_pool
  --dual_head_soft_pool_spatial_factor 5
  --dual_tail_codec_type p_delta
  --dual_tail_p_delta_spatial_factor 4
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  INFER_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

echo "[pipeline] Step 1/2: recover infer"
torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${INFER_ARGS[@]}" \
  "$@"

RUN_DECODER="${RUN_DECODER:-1}"
DECODER_DEVICE="${DECODER_DEVICE:-cuda}"
DECODER_COMPARE_MODE="${DECODER_COMPARE_MODE:-both}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-${PROJECT_ROOT}/reconstruct/videos}"

if [[ "${RUN_DECODER}" == "1" ]]; then
  echo "[pipeline] Step 2/2: decode recover latents to videos"
  python "${PROJECT_ROOT}/reconstruct/decoder.py" \
    -R "${OUTPUT_DIR}" \
    --device "${DECODER_DEVICE}" \
    --compare_mode "${DECODER_COMPARE_MODE}" \
    --source_video_dir "${SOURCE_VIDEO_DIR}"
else
  echo "[pipeline] Decoder skipped because RUN_DECODER=${RUN_DECODER}"
fi
