#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Optional environment activation for bare shell sessions.
if [[ -f /data/heli/miniconda3/bin/activate ]]; then
    source /data/heli/miniconda3/bin/activate teleai || true
fi

cd "${PROJECT_ROOT}"

# Volcano usually injects VC_* env vars. Keep manual overrides possible.
export NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-8}}"
export NNODES="${NNODES:-$(( ${VC_MASTER_NUM:-1} + ${VC_WORKER_NUM:-0} ))}"
export NODE_RANK="${NODE_RANK:-${RANK:-${VC_TASK_INDEX:-0}}}"

# VC_MASTER_HOSTS may contain multiple hosts separated by comma.
MASTER_ADDR_FROM_VC="${VC_MASTER_HOSTS:-127.0.0.1}"
MASTER_ADDR_FROM_VC="${MASTER_ADDR_FROM_VC%%,*}"
export MASTER_ADDR="${MASTER_ADDR:-${MASTER_ADDR_FROM_VC}}"
export MASTER_PORT="${MASTER_PORT:-12445}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export NUM_MACHINES="${NUM_MACHINES:-${NNODES}}"
export MACHINE_RANK="${MACHINE_RANK:-${NODE_RANK}}"
export NUM_PROCESSES_PER_MACHINE="${NUM_PROCESSES_PER_MACHINE:-${NPROC_PER_NODE}}"
export NUM_PROCESSES="$((NUM_MACHINES * NUM_PROCESSES_PER_MACHINE))"

export ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${PROJECT_ROOT}/scripts/accelerate_configs/multi_node_example_zero3.yaml}"
export RECOVER_INPUT_PATH="${RECOVER_INPUT_PATH:-${PROJECT_ROOT}/reconstruct/latents}"
export RECOVER_OUTPUT_DIR="${RECOVER_OUTPUT_DIR:-${PROJECT_ROOT}/reconstruct/gen2recon_runs_CNN_3}"
export HELIOS_BASE_MODEL_PATH="${HELIOS_BASE_MODEL_PATH:-/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base}"

# NCCL knobs for multi-node stability/tuning.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

if [[ ! -f "${ACCELERATE_CONFIG_FILE}" ]]; then
    echo "[ERROR] Accelerate config not found: ${ACCELERATE_CONFIG_FILE}" >&2
    exit 1
fi

if [[ ! -e "${RECOVER_INPUT_PATH}" ]]; then
    echo "[ERROR] recover train input_path does not exist: ${RECOVER_INPUT_PATH}" >&2
    echo "[HINT] Prepare latent .pt files first, for example:" >&2
    echo "  cd ${PROJECT_ROOT} && python reconstruct/encoder.py --input_dir <video_dir> --output_dir ${PROJECT_ROOT}/reconstruct/latents" >&2
    exit 1
fi

BASE_MODEL_ARGS=()
if [[ -n "${HELIOS_BASE_MODEL_PATH}" ]]; then
    if [[ ! -d "${HELIOS_BASE_MODEL_PATH}" ]]; then
        echo "[ERROR] HELIOS_BASE_MODEL_PATH does not exist: ${HELIOS_BASE_MODEL_PATH}" >&2
        exit 1
    fi
    BASE_MODEL_ARGS+=(--base_model_path "${HELIOS_BASE_MODEL_PATH}")
fi

mkdir -p "${RECOVER_OUTPUT_DIR}"

echo "[INFO] MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} NNODES=${NNODES} NODE_RANK=${NODE_RANK} NPROC_PER_NODE=${NPROC_PER_NODE}" >&2
echo "[INFO] INPUT=${RECOVER_INPUT_PATH} OUTPUT=${RECOVER_OUTPUT_DIR}" >&2

accelerate launch \
    --config_file "${ACCELERATE_CONFIG_FILE}" \
    --num_machines "${NUM_MACHINES}" \
    --machine_rank "${MACHINE_RANK}" \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_ip "${MASTER_ADDR}" \
    --main_process_port "${MASTER_PORT}" \
    "${PROJECT_ROOT}/reconstruct/recover.py" train \
    --input_path "${RECOVER_INPUT_PATH}" \
    --output_dir "${RECOVER_OUTPUT_DIR}" \
    "${BASE_MODEL_ARGS[@]}" \
    "$@"