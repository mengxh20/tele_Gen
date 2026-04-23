#!/bin/bash
set -euo pipefail

# Recover 14B full fine-tuning should use Accelerate + DeepSpeed ZeRO-3 so model states are sharded.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29521}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export NUM_MACHINES="${NUM_MACHINES:-${NNODES}}"
export MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK}}"
export NUM_PROCESSES_PER_MACHINE="${NUM_PROCESSES_PER_MACHINE:-${NPROC_PER_NODE}}"
export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-scripts/accelerate_configs/multi_node_example_zero3.yaml}"
export NUM_PROCESSES="$((NUM_MACHINES * NUM_PROCESSES_PER_MACHINE))"

# Aggressive low-bitrate defaults. Override any of these from the shell when needed.
export TEMPORAL_FACTOR="${TEMPORAL_FACTOR:-4}"
export SPATIAL_FACTOR="${SPATIAL_FACTOR:-8}"
export SECTION_SPAN_LATENTS="${SECTION_SPAN_LATENTS:-5}"
export ANCHOR_SPAN_LATENTS="${ANCHOR_SPAN_LATENTS:-1}"
export ANCHOR_SPATIAL_FACTOR="${ANCHOR_SPATIAL_FACTOR:-2}"
export KEYFRAME_CODEC_MODE="${KEYFRAME_CODEC_MODE:-quantized_int8}"
export KEYFRAME_QUANT_DTYPE="${KEYFRAME_QUANT_DTYPE:-int8}"
export KEYFRAME_SPATIAL_FACTOR="${KEYFRAME_SPATIAL_FACTOR:-2}"
export RATE_LOSS_WEIGHT="${RATE_LOSS_WEIGHT:-2.0}"
export RATE_LOSS_WARMUP_STEPS="${RATE_LOSS_WARMUP_STEPS:-0}"
export RATE_LOSS_RAMP_STEPS="${RATE_LOSS_RAMP_STEPS:-0}"
export ENTROPY_AUX_LEARNING_RATE="${ENTROPY_AUX_LEARNING_RATE:-1e-3}"

accelerate launch \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    reconstruct/recover.py train \
    --output_dir "reconstruct/gen2recon_runs_HE2E_2" \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 500 \
    --temporal_factor "${TEMPORAL_FACTOR}" \
    --spatial_factor "${SPATIAL_FACTOR}" \
    --latent_window_size "${SECTION_SPAN_LATENTS}" \
    --section_span_latents "${SECTION_SPAN_LATENTS}" \
    --anchor_span_latents "${ANCHOR_SPAN_LATENTS}" \
    --anchor_spatial_factor "${ANCHOR_SPATIAL_FACTOR}" \
    --keyframe_codec_mode "${KEYFRAME_CODEC_MODE}" \
    --keyframe_quant_dtype "${KEYFRAME_QUANT_DTYPE}" \
    --keyframe_spatial_factor "${KEYFRAME_SPATIAL_FACTOR}" \
    --rate_loss_weight "${RATE_LOSS_WEIGHT}" \
    --rate_loss_warmup_steps "${RATE_LOSS_WARMUP_STEPS}" \
    --rate_loss_ramp_steps "${RATE_LOSS_RAMP_STEPS}" \
    --entropy_aux_learning_rate "${ENTROPY_AUX_LEARNING_RATE}" \
    "$@"
