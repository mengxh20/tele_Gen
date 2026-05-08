#!/bin/bash
set -euo pipefail


echo ">>>> Start Time: $(date +"%Y-%m-%d %H:%M:%S.%N") <<<<"

cd /app1/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/tele_Gen

source /app1/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/miniconda3/bin/activate
conda activate helios

export NCCL_ALGO=RING
# export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=300000

NNODES=$((${VC_MASTER_NUM:-1} + ${VC_WORKER_NUM:-0}))
MASTER_ADDR=${VC_MASTER_HOSTS:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-12445}
NODE_RANK=${RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Recover 14B full fine-tuning should use Accelerate + DeepSpeed ZeRO-3 so model states are sharded.
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
# export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# export NNODES="${NNODES:-1}"
# export NODE_RANK="${NODE_RANK:-0}"
# export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# export MASTER_PORT="${MASTER_PORT:-29526}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
# export NUM_MACHINES="${NUM_MACHINES:-${NNODES}}"
# export MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK}}"
# export NUM_PROCESSES_PER_MACHINE="${NUM_PROCESSES_PER_MACHINE:-${NPROC_PER_NODE}}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/multi_node_example_zero3.yaml}"
# export NUM_PROCESSES="$((NUM_MACHINES * NUM_PROCESSES_PER_MACHINE))"
export SAVEDIR="${SAVEDIR:-reconstruct/gen2recon_runs_HE2E_May06}"

# Quality-first defaults. Override any of these from the shell when bitrate pressure matters more.
export TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-2}"
export SPATIAL_FACTOR="${SPATIAL_FACTOR:-4}"
export SECTION_SPAN_LATENTS="${SECTION_SPAN_LATENTS:-4}"
export RECOVER_OVERLAP_LATENTS="${RECOVER_OVERLAP_LATENTS:-2}"
export ANCHOR_SPAN_LATENTS="${ANCHOR_SPAN_LATENTS:-1}"
export ANCHOR_SPATIAL_FACTOR="${ANCHOR_SPATIAL_FACTOR:-1}"
export KEYFRAME_CODEC_MODE="${KEYFRAME_CODEC_MODE:-quantized_int8}"
export KEYFRAME_QUANT_DTYPE="${KEYFRAME_QUANT_DTYPE:-int8}"
export KEYFRAME_SPATIAL_FACTOR="${KEYFRAME_SPATIAL_FACTOR:-2}"
export RATE_LOSS_WEIGHT="${RATE_LOSS_WEIGHT:-0.5}"
export RATE_LOSS_WARMUP_STEPS="${RATE_LOSS_WARMUP_STEPS:-0}"
export RATE_LOSS_RAMP_STEPS="${RATE_LOSS_RAMP_STEPS:-1000}"
export ENTROPY_AUX_LEARNING_RATE="${ENTROPY_AUX_LEARNING_RATE:-1e-3}"
export TEMPORAL_DELTA_LOSS_WEIGHT="${TEMPORAL_DELTA_LOSS_WEIGHT:-1.0}"
export MOTION_LOSS_MAX_WEIGHT="${MOTION_LOSS_MAX_WEIGHT:-2.0}"
export MOTION_LOSS_GAMMA="${MOTION_LOSS_GAMMA:-1.0}"
export DYNAMIC_RATE_ENABLED="${DYNAMIC_RATE_ENABLED:-true}"
export MOTION_SCORE_TYPE="${MOTION_SCORE_TYPE:-latent_delta_l1_norm}"
export TAIL_QUALITY_MIN="${TAIL_QUALITY_MIN:-0.9}"
export TAIL_QUALITY_MAX="${TAIL_QUALITY_MAX:-1.4}"
export ANCHOR_PROFILE_LOW="${ANCHOR_PROFILE_LOW:-3}"
export ANCHOR_PROFILE_BASE="${ANCHOR_PROFILE_BASE:-2}"
export ANCHOR_PROFILE_HIGH="${ANCHOR_PROFILE_HIGH:-2}"

DYNAMIC_RATE_FLAG="--dynamic_rate_enabled"
if [[ "${DYNAMIC_RATE_ENABLED}" == "0" || "${DYNAMIC_RATE_ENABLED}" == "false" || "${DYNAMIC_RATE_ENABLED}" == "False" ]]; then
    DYNAMIC_RATE_FLAG="--no-dynamic_rate_enabled"
fi

ACCELERATE_SETTINGS=(
    --num_nodes "$NNODES"
    --num_processes $(($NNODES * $GPUS_PER_NODE))  # 注意：这里是总进程数（总GPU数）
    --node_rank "$NODE_RANK"
    --main_process_ip "$MASTER_ADDR"
    --main_process_port "$MASTER_PORT"
    --config_file "${ACCELERATE_CONFIG_FILE}"
    --multi_gpu                                    # 开启多卡支持
)

accelerate launch \
    "${ACCELERATE_SETTINGS[@]}" \
    reconstruct/recover.py train \
    --output_dir "${SAVEDIR}" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 2 \
    --max_steps 1000 \
    --temporal_factor "${TEMPORAL_FACTOR}" \
    --spatial_factor "${SPATIAL_FACTOR}" \
    --latent_window_size "${SECTION_SPAN_LATENTS}" \
    --section_span_latents "${SECTION_SPAN_LATENTS}" \
    --history_sizes None None 3 \
    --recover_overlap_latents "${RECOVER_OVERLAP_LATENTS}" \
    --anchor_span_latents "${ANCHOR_SPAN_LATENTS}" \
    --anchor_spatial_factor "${ANCHOR_SPATIAL_FACTOR}" \
    --keyframe_codec_mode "${KEYFRAME_CODEC_MODE}" \
    --keyframe_quant_dtype "${KEYFRAME_QUANT_DTYPE}" \
    --keyframe_spatial_factor "${KEYFRAME_SPATIAL_FACTOR}" \
    --rate_loss_weight "${RATE_LOSS_WEIGHT}" \
    --rate_loss_warmup_steps "${RATE_LOSS_WARMUP_STEPS}" \
    --rate_loss_ramp_steps "${RATE_LOSS_RAMP_STEPS}" \
    --entropy_aux_learning_rate "${ENTROPY_AUX_LEARNING_RATE}" \
    --temporal_delta_loss_weight "${TEMPORAL_DELTA_LOSS_WEIGHT}" \
    --motion_loss_max_weight "${MOTION_LOSS_MAX_WEIGHT}" \
    --motion_loss_gamma "${MOTION_LOSS_GAMMA}" \
    "${DYNAMIC_RATE_FLAG}" \
    --motion_score_type "${MOTION_SCORE_TYPE}" \
    --tail_quality_min "${TAIL_QUALITY_MIN}" \
    --tail_quality_max "${TAIL_QUALITY_MAX}" \
    --anchor_profile_low "${ANCHOR_PROFILE_LOW}" \
    --anchor_profile_base "${ANCHOR_PROFILE_BASE}" \
    --anchor_profile_high "${ANCHOR_PROFILE_HIGH}" \
    "$@"

echo ">>>> End Time: $(date +"%Y-%m-%d %H:%M:%S.%N") <<<<"