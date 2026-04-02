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

accelerate launch \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    reconstruct/recover.py train "$@"
