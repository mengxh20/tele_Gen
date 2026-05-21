#!/usr/bin/env bash
set -euo pipefail

cd /ddn/team/shared/heli/code/tele_Gen

export PATH=/ddn/team/shared/heli/miniconda3/envs/teleai/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

export MASTER_ADDR="${MASTER_ADDR:-10.11.1.109}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export NNODES="${NNODES:-2}"
export NODE_RANK=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth2}"

# Conservative TCP-only NCCL test settings. This avoids IB/RDMA and small /dev/shm issues
# while checking that torchrun + NCCL can communicate across the two hosts.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

export NCCL_TEST_SCRIPT="${NCCL_TEST_SCRIPT:-/tmp/test_multinode_nccl.py}"
export NCCL_TEST_TENSOR_MB="${NCCL_TEST_TENSOR_MB:-16}"
export NCCL_TEST_ITERS="${NCCL_TEST_ITERS:-5}"

cat > "${NCCL_TEST_SCRIPT}" <<'PY'
import os
import socket
import time

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    tensor_mb = int(os.environ.get("NCCL_TEST_TENSOR_MB", "16"))
    iters = int(os.environ.get("NCCL_TEST_ITERS", "5"))

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    numel = max(1, tensor_mb * 1024 * 1024 // 4)
    expected = world_size * (world_size + 1) / 2.0
    x = torch.empty(numel, device="cuda", dtype=torch.float32)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        x.fill_(rank + 1)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    got = float(x[0].item())
    if abs(got - expected) > 1e-3:
        raise RuntimeError(f"rank={rank}: all_reduce got {got}, expected {expected}")

    host = socket.gethostname()
    print(
        f"[host={host}] rank={rank}/{world_size} local_rank={local_rank} "
        f"cuda={torch.cuda.current_device()} tensor_mb={tensor_mb} "
        f"iters={iters} all_reduce_sum={got} elapsed={elapsed:.3f}s",
        flush=True,
    )

    dist.barrier()
    if rank == 0:
        print("[OK] multinode NCCL all_reduce passed", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
PY

echo "[INFO] role=worker addr=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} rank=${NODE_RANK}"
echo "[INFO] nproc_per_node=${NPROC_PER_NODE} cuda_visible_devices=${CUDA_VISIBLE_DEVICES} nccl_ifname=${NCCL_SOCKET_IFNAME}"
echo "[INFO] tensor_mb=${NCCL_TEST_TENSOR_MB} iters=${NCCL_TEST_ITERS} ib_disable=${NCCL_IB_DISABLE} shm_disable=${NCCL_SHM_DISABLE}"

torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${NCCL_TEST_SCRIPT}"
