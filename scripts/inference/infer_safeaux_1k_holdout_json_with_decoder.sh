#!/usr/bin/env bash
set -euo pipefail

cd /data/heli/code/tele_Gen

export PATH=/data/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6}"

DDN_MODEL=/ddn/team/shared/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base
DATA_MODEL=/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base

if [[ -z "${HELIOS_BASE_MODEL_PATH:-}" ]]; then
  if [[ -d "${DDN_MODEL}" ]]; then
    export HELIOS_BASE_MODEL_PATH="${DDN_MODEL}"
  else
    export HELIOS_BASE_MODEL_PATH="${DATA_MODEL}"
  fi
fi

INPUT_PATH="${INPUT_PATH:-/data/heli/code/tele_Gen/reconstruct/latents_nochunk_149_holdout_json}"
SOURCE_VIDEO_DIR="${SOURCE_VIDEO_DIR:-/data/heli/code/tele_Gen/reconstruct/videos_149_holdout_json}"

BASE_RUN_DIR=/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_safeaux_base_1k_section7_anchor1_steps24
EP120_RUN_DIR=/data/heli/code/tele_Gen/reconstruct/experiments/recover_sparse_refresh_1w_x0int8_headsoft5_tailp4_safeaux_ep120_1k_section7_anchor1_steps24

run_one() {
  local label="$1"
  local run_dir="$2"
  local ckpt="${run_dir}/checkpoints/epoch_0001_step_00001000"
  local out_dir="${run_dir}/recover_outputs_holdout_json_step1000_x0int8sf1_headsoft5_tailp4"

  echo "[INFO] label=${label}"
  echo "[INFO] checkpoint=${ckpt}"
  echo "[INFO] output=${out_dir}"

  torchrun --nproc_per_node=1 reconstruct/recover.py infer \
    --input_path "${INPUT_PATH}" \
    --checkpoint_dir "${ckpt}" \
    --output_dir "${out_dir}" \
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
    --soft_tail_hint_step_fraction 0.5

  python reconstruct/decoder.py \
    -R "${out_dir}" \
    --source_video_dir "${SOURCE_VIDEO_DIR}" \
    --base_model_path "${HELIOS_BASE_MODEL_PATH}" \
    --compute_lpips \
    --lpips_batch_size 8
}

echo "[INFO] base_model=${HELIOS_BASE_MODEL_PATH}"
echo "[INFO] input=${INPUT_PATH}"
echo "[INFO] source_video_dir=${SOURCE_VIDEO_DIR}"

run_one "base_1k" "${BASE_RUN_DIR}"
run_one "ep120_1k" "${EP120_RUN_DIR}"
