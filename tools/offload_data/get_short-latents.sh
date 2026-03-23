export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
#################################################################
## Torch
#################################################################
export TOKENIZERS_PARALLELISM=false
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
#################################################################


#################################################################
## NCCL
#################################################################
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0  # was 1
export NCCL_SHM_DISABLE=0  # was 1
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
#################################################################

#################################################################
## DIST
#################################################################
# MASTER_ADDR=$ARNOLD_WORKER_0_HOST
# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# MASTER_PORT=${ports[0]}
# NUM_MACHINES=$ARNOLD_WORKER_NUM
# MACHINE_RANK=$ARNOLD_ID
# NUM_PROCESSES_PER_MACHINE=$ARNOLD_WORKER_GPU
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}      # 或 localhost
export MASTER_PORT=${MASTER_PORT:-29500}          # 任意一个没被占用的端口
export NUM_MACHINES=${NUM_MACHINES:-1}
export MACHINE_RANK=${MACHINE_RANK:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export NUM_PROCESSES_PER_MACHINE=${NUM_PROCESSES_PER_MACHINE:-1}

# export CUDA_VISIBLE_DEVICES=1
# export MASTER_PORT=12345
# export NUM_PROCESSES_PER_MACHINE=1
# export NUM_MACHINES=1
# export MACHINE_RANK=0

VISIBLE_GPU_COUNT=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | sed '/^[[:space:]]*$/d' | wc -l)
if [ "${NUM_PROCESSES_PER_MACHINE}" -gt "${VISIBLE_GPU_COUNT}" ]; then
    echo "NUM_PROCESSES_PER_MACHINE (${NUM_PROCESSES_PER_MACHINE}) exceeds visible GPUs (${VISIBLE_GPU_COUNT})."
    exit 1
fi

WORLD_SIZE=$((NUM_PROCESSES_PER_MACHINE * NUM_MACHINES))
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_PROCESSES_PER_MACHINE} --nnodes ${NUM_MACHINES} --node_rank ${MACHINE_RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
if [ -n "${RDZV_BACKEND}" ]; then
    DISTRIBUTED_ARGS="${DISTRIBUTED_ARGS} --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} --rdzv_id 9863 --rdzv_backend ${RDZV_BACKEND}"
    export NCCL_SHM_DISABLE=1
fi

echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################
# 
torchrun $DISTRIBUTED_ARGS \
    tools/offload_data/get_short-latents.py
