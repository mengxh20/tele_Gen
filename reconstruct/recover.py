"""
recover.py 是压缩与恢复主线的核心入口，负责两类工作：

1. `train`:
   基于原始 clean latents 与低码率 low latents 的差异，训练一个 stage1 recover transformer，
   让模型在只看到首帧和有限历史上下文的条件下，恢复后续 latent 细节。
2. `infer`:
   先把输入 latent 序列压缩为 low_latents，再调用训练好的 recover transformer 进行重建，
   最终输出 low_latents、recover_latents 和指标，而不是直接生成视频。

整体设计尽量复用 Helios 已有的 stage1 / transformer / scheduler 能力，把新增逻辑收敛在
`reconstruct/` 下，通过“低码率编码 + 条件恢复”的方式服务生成式视频压缩与恢复链路。
"""

import argparse
import json
import math
import os
import random
import sys
import types
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from reconstruct.latent_io import (
    DEFAULT_BASE_MODEL_PATH,
    LATENT_FORMAT_V2,
    LOW_LATENT_FORMAT_V1,
    build_chunk_frame_ranges,
    compute_bpp_from_total_pixels,
    flatten_latent_chunks,
    infer_source_num_frames,
    load_payload,
    resolve_device,
    resolve_source_resolution,
    split_full_latents,
    validate_payload,
)


DEFAULT_INPUT_PATH = Path("reconstruct/latents")
DEFAULT_TRAIN_OUTPUT_DIR = Path("reconstruct/gen2recon_runs")
DEFAULT_INFER_OUTPUT_DIR = Path("reconstruct/recover_outputs")
DEFAULT_TEMPORAL_FACTOR = 2
DEFAULT_SPATIAL_FACTOR = 2
DEFAULT_QUANT_DTYPE = "int8"
DEFAULT_KEYFRAME_DTYPE = "float16"
DEFAULT_HISTORY_SIZES = [16, 2, 1]
DEFAULT_LATENT_WINDOW_SIZE = 9
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_WEIGHT_DTYPE = "bf16"
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_LOSS_WEIGHTING_SCHEME = "logit_normal"
DEFAULT_LOGIT_MEAN = 0.0
DEFAULT_LOGIT_STD = 1.0
DEFAULT_MODE_SCALE = 1.29
DEFAULT_LOSS_EMA_DECAY = 0.95
DEFAULT_FLOW_LOSS_WEIGHT = 1.0
DEFAULT_X_LOSS_WEIGHT = 0.25
DEFAULT_NOISE_LOSS_WEIGHT = 0.25
DEFAULT_AUX_LOSS_WARMUP_STEPS = 500
DEFAULT_AUX_LOSS_RAMP_STEPS = 1000
RECOVER_CONFIG_VERSION = "helios_recover_v1"
EPS = 1e-8


@dataclass
class CodecConfig:
    """低码率 latent 编码配置。

    这里的压缩策略比较直接：
    - 首帧 `keyframe` 直接保留较高精度，作为恢复过程中的稳定锚点。
    - 其余帧 `remainder` 做时空下采样，再按通道量化，模拟低码率传输结果。
    """

    temporal_factor: int = DEFAULT_TEMPORAL_FACTOR
    spatial_factor: int = DEFAULT_SPATIAL_FACTOR
    quant_dtype: str = DEFAULT_QUANT_DTYPE
    keyframe_dtype: str = DEFAULT_KEYFRAME_DTYPE


@dataclass
class SequenceMetadata:
    """描述一段 latent 序列及其原始视频统计信息。

    这些字段既用于恢复后重新打包 latent chunks，也用于计算 raw_bpp、low_bpp 等码率指标，
    方便评估低码率压缩与恢复链路的收益。
    """

    source_fps: float
    source_num_frames: int
    source_width: int
    source_height: int
    total_pixels: int
    raw_file_bytes: int
    raw_bpp: float
    chunk_lengths: List[int]
    chunk_frame_ranges: List[Tuple[int, int]]


@dataclass
class PreparedSequence:
    """把单个输入样本预处理成训练 / 推理都可直接使用的结构。

    一个样本同时保留：
    - `clean_full_latents`: 原始高保真 latent，作为训练监督和推理对照真值。
    - `low_codec_payload`: 低码率编码后的中间载荷，可视为传输结果。
    - `low_full_latents`: 从低码率载荷直接解码出的粗恢复 latent，用于构造历史条件。
    """

    path: Path
    metadata: SequenceMetadata
    clean_full_latents: torch.Tensor
    low_codec_payload: Dict[str, object]
    low_full_latents: torch.Tensor


@dataclass(frozen=True)
class DistributedContext:
    """推理阶段的分布式上下文。

    这里主要服务 `infer` 命令：把不同 section 的恢复任务分摊到多个 rank，
    最后只在主进程汇总重建结果并落盘。
    """

    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass
class TrainingStepOutput:
    """封装单个优化步中最关键的训练统计量。"""

    loss: torch.Tensor
    raw_loss: torch.Tensor
    flow_loss: torch.Tensor
    x_loss: torch.Tensor
    noise_loss: torch.Tensor
    aux_scale: torch.Tensor
    sigma_mean: torch.Tensor
    flow_target_rms: torch.Tensor
    flow_pred_rms: torch.Tensor
    x_target_rms: torch.Tensor
    x_pred_rms: torch.Tensor
    noise_target_rms: torch.Tensor
    noise_pred_rms: torch.Tensor

    def detached_float_metrics(self) -> Dict[str, torch.Tensor]:
        """把日志指标转成脱离计算图的 float tensor，便于跨进程聚合与记录。"""
        return {
            "loss": self.loss.detach().float(),
            "raw_loss": self.raw_loss.detach().float(),
            "flow_loss": self.flow_loss.detach().float(),
            "x_loss": self.x_loss.detach().float(),
            "noise_loss": self.noise_loss.detach().float(),
            "aux_scale": self.aux_scale.detach().float(),
            "sigma_mean": self.sigma_mean.detach().float(),
            "flow_target_rms": self.flow_target_rms.detach().float(),
            "flow_pred_rms": self.flow_pred_rms.detach().float(),
            "x_target_rms": self.x_target_rms.detach().float(),
            "x_pred_rms": self.x_pred_rms.detach().float(),
            "noise_target_rms": self.noise_target_rms.detach().float(),
            "noise_pred_rms": self.noise_pred_rms.detach().float(),
        }


class LatentWindowDataset(Dataset):
    """把整段 latent 序列切成“历史窗口 + 目标窗口”的训练样本。

    训练 recover transformer 的目标不是一次恢复整段视频，而是学习：
    - 给定首帧 `x0`
    - 给定低码率解码后的历史 latent
    - 恢复当前 section 的高保真 latent

    这种按窗口训练的方式更贴近真实链路中的分段恢复过程，也能降低显存压力。
    """

    def __init__(
        self,
        sequences: Sequence[PreparedSequence],
        history_sizes: Sequence[int],
        latent_window_size: int,
    ):
        self.sequences = list(sequences)
        self.history_sizes = list(history_sizes)
        self.history_window_size = sum(history_sizes)
        self.latent_window_size = latent_window_size
        self.samples: List[Tuple[int, int]] = []

        for seq_idx, sequence in enumerate(self.sequences):
            total_latent_frames = sequence.clean_full_latents.shape[1]
            # section_start 从 1 开始，意味着首帧默认单独保留为 keyframe，
            # 后续所有帧都交给 recover 模块按窗口学习恢复。
            for start in range(1, total_latent_frames, latent_window_size):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        seq_idx, section_start = self.samples[index]
        sequence = self.sequences[seq_idx]

        # 目标窗口来自 clean latents，作为监督信号；
        # 历史窗口来自 low latents，模拟接收端在低码率条件下可见的信息。
        target_latents, valid_target_frames = extract_target_window(
            clean_full_latents=sequence.clean_full_latents,
            section_start=section_start,
            latent_window_size=self.latent_window_size,
        )
        history_latents = extract_history_window(
            low_full_latents=sequence.low_full_latents,
            section_start=section_start,
            history_window_size=self.history_window_size,
        )

        return {
            "history_latents": history_latents,
            "target_latents": target_latents,
            "x0_latents": sequence.clean_full_latents[:, :1],
            "valid_target_frames": valid_target_frames,
        }


def parse_args() -> argparse.Namespace:
    """解析命令行，统一承载训练与推理两个子命令。"""
    parser = argparse.ArgumentParser(description="Train and run Helios latent recovery for generative compression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the latent recovery network from ori_latents.")
    add_common_train_args(train_parser)

    infer_parser = subparsers.add_parser("infer", help="Compress ori_latents and reconstruct recover_latents.")
    add_common_infer_args(infer_parser)

    return parser.parse_args()


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    """注册训练阶段使用的参数。"""
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--weight_dtype", type=str, default=DEFAULT_WEIGHT_DTYPE, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_factor", type=int, default=DEFAULT_TEMPORAL_FACTOR)
    parser.add_argument("--spatial_factor", type=int, default=DEFAULT_SPATIAL_FACTOR)
    parser.add_argument("--quant_dtype", type=str, default=DEFAULT_QUANT_DTYPE, choices=["int8"])
    parser.add_argument("--keyframe_dtype", type=str, default=DEFAULT_KEYFRAME_DTYPE, choices=["float16", "float32"])
    parser.add_argument("--history_sizes", type=int, nargs="+", default=DEFAULT_HISTORY_SIZES)
    parser.add_argument("--latent_window_size", type=int, default=DEFAULT_LATENT_WINDOW_SIZE)
    parser.add_argument(
        "--loss_weighting_scheme",
        type=str,
        default=DEFAULT_LOSS_WEIGHTING_SCHEME,
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help="Helios-aligned loss weighting and timestep sampling scheme for stage1 recover training.",
    )
    parser.add_argument("--logit_mean", type=float, default=DEFAULT_LOGIT_MEAN)
    parser.add_argument("--logit_std", type=float, default=DEFAULT_LOGIT_STD)
    parser.add_argument("--mode_scale", type=float, default=DEFAULT_MODE_SCALE)
    parser.add_argument(
        "--loss_ema_decay",
        type=float,
        default=DEFAULT_LOSS_EMA_DECAY,
        help="EMA decay used only for smoother loss display and logging.",
    )
    parser.add_argument("--flow_loss_weight", type=float, default=DEFAULT_FLOW_LOSS_WEIGHT)
    parser.add_argument("--x_loss_weight", type=float, default=DEFAULT_X_LOSS_WEIGHT)
    parser.add_argument("--noise_loss_weight", type=float, default=DEFAULT_NOISE_LOSS_WEIGHT)
    parser.add_argument("--aux_loss_warmup_steps", type=int, default=DEFAULT_AUX_LOSS_WARMUP_STEPS)
    parser.add_argument("--aux_loss_ramp_steps", type=int, default=DEFAULT_AUX_LOSS_RAMP_STEPS)


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    """注册推理阶段使用的参数。"""
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_INFER_OUTPUT_DIR)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default=DEFAULT_WEIGHT_DTYPE,
        choices=["bf16", "fp16", "fp32"],
        help="Weight dtype used when loading the trained recovery transformer.",
    )
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)


def main() -> None:
    """按子命令路由到训练或推理主流程。"""
    args = parse_args()
    if args.command == "train":
        command_train(args)
    elif args.command == "infer":
        command_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def ensure_diffusers_parallel_shim() -> None:
    """给当前环境补一个最小 shim，兼容 Helios 依赖的 diffusers 并行接口。

    某些环境里 `diffusers.models._modeling_parallel` 不存在，但 Helios 的导入链会引用它。
    这里提供一个轻量替身，避免因为环境细节导致 recover 主链路无法启动。
    """
    try:
        import diffusers.models._modeling_parallel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    shim_module = types.ModuleType("diffusers.models._modeling_parallel")

    class ContextParallelInput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ContextParallelOutput(ContextParallelInput):
        pass

    shim_module.ContextParallelInput = ContextParallelInput
    shim_module.ContextParallelOutput = ContextParallelOutput
    sys.modules["diffusers.models._modeling_parallel"] = shim_module


def import_helios_inference_stack():
    """延迟导入 Helios 推理栈。

    这样做有两个目的：
    - 减少模块 import 时的环境耦合，只有真正训练 / 推理时才要求 Helios 依赖完整。
    - 把 recover 逻辑尽量封装在 `reconstruct/`，仅在必要时调用 Helios 底座能力。
    """
    ensure_diffusers_parallel_shim()
    try:
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
        from helios.utils.utils_base import encode_prompt
    except ImportError as exc:
        raise ImportError(
            "Failed to import the Helios transformer stack. "
            "Please use the project's intended diffusers environment before running recover.py."
        ) from exc

    return HeliosTransformer3DModel, HeliosScheduler, encode_prompt


def import_stage1_prepare_fn():
    """导入 Helios 的 stage1 条件拼装函数。

    recover 模块复用它来组织：
    - 当前目标 latent
    - 首帧 x0
    - 多尺度历史 latent

    这样可以最大程度复用 Helios 现有输入格式，而不是重写底层拼接逻辑。
    """
    from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents

    return prepare_stage1_clean_input_from_latents


def parse_accelerator_mixed_precision(weight_dtype: str) -> str:
    """把项目里的 dtype 字符串映射成 accelerate 所需的 mixed_precision 取值。"""
    if weight_dtype == "bf16":
        return "bf16"
    if weight_dtype == "fp16":
        return "fp16"
    if weight_dtype == "fp32":
        return "no"
    raise ValueError(f"Unsupported mixed precision weight dtype: {weight_dtype}")


def import_diffusers_training_utils():
    """导入 diffusers 中与 sigma 采样和 loss 加权相关的工具函数。"""
    try:
        from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
    except ImportError as exc:
        raise ImportError(
            "Recover training requires diffusers.training_utils. "
            "Please activate the project's intended diffusers environment before running recover.py."
        ) from exc

    return compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


def validate_train_args(args: argparse.Namespace) -> None:
    """校验训练参数，避免日志平滑等辅助参数取值异常。"""
    if not 0.0 <= args.loss_ema_decay < 1.0:
        raise ValueError(f"loss_ema_decay must be in [0, 1), got {args.loss_ema_decay}.")
    for key in ("flow_loss_weight", "x_loss_weight", "noise_loss_weight"):
        value = getattr(args, key)
        if value < 0.0:
            raise ValueError(f"{key} must be >= 0, got {value}.")
    if args.aux_loss_warmup_steps < 0:
        raise ValueError(f"aux_loss_warmup_steps must be >= 0, got {args.aux_loss_warmup_steps}.")
    if args.aux_loss_ramp_steps < 0:
        raise ValueError(f"aux_loss_ramp_steps must be >= 0, got {args.aux_loss_ramp_steps}.")


def create_train_accelerator(weight_dtype: str, gradient_accumulation_steps: int):
    """构造 accelerate 的 Accelerator。

    recover 训练默认支持多卡和混合精度，这里统一处理：
    - 分布式通信 backend
    - 梯度累积
    - bf16 / fp16 / fp32 的混合精度策略
    """
    try:
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
    except ImportError as exc:
        raise ImportError(
            "Recover training requires accelerate. Please activate the project's intended training environment."
        ) from exc

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs = InitProcessGroupKwargs(backend=backend, timeout=timedelta(seconds=1800))
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=parse_accelerator_mixed_precision(weight_dtype),
        kwargs_handlers=[init_kwargs],
    )


def init_offline_wandb_run(
    output_dir: Path,
    args: argparse.Namespace,
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    checkpoint_save_interval_epochs: int,
):
    """初始化离线 wandb run。

    这里不依赖在线服务，主要作用是把训练损失、sigma、RMS 等指标持久化，
    方便后续分析恢复质量与训练稳定性。
    """
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("Recover training requires wandb for offline loss logging.") from exc

    os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(
        project="helios-recover",
        job_type="train",
        name=output_dir.name,
        mode="offline",
        dir=str(output_dir),
        config={
            "input_path": str(args.input_path.resolve()),
            "output_dir": str(output_dir),
            "base_model_path": args.base_model_path,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "weight_dtype": args.weight_dtype,
            "history_sizes": list(history_sizes),
            "latent_window_size": latent_window_size,
            "codec_config": asdict(codec_config),
            "checkpoint_save_interval_epochs": checkpoint_save_interval_epochs,
            "flow_loss_weight": args.flow_loss_weight,
            "x_loss_weight": args.x_loss_weight,
            "noise_loss_weight": args.noise_loss_weight,
            "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
            "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        },
        reinit=True,
    )


def build_periodic_checkpoint_dir(output_dir: Path, epoch: int, global_step: int) -> Path:
    """为阶段性 checkpoint 生成稳定目录名，便于恢复与对比不同训练阶段效果。"""
    return output_dir / "checkpoints" / f"epoch_{epoch:04d}_step_{global_step:08d}"


def resolve_train_device(device_arg: str) -> torch.device:
    """解析训练设备。

    单卡时直接复用 `latent_io.resolve_device`；
    多卡时强制跟随 launcher 设置的 `LOCAL_RANK`，确保每个进程只绑定自己的 GPU。
    """
    requested_device = torch.device(device_arg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_device.type != "cuda" or world_size <= 1:
        return resolve_device(device_arg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is not available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("Accelerate training requires LOCAL_RANK to be set by the launcher.")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


def command_train(args: argparse.Namespace) -> None:
    """训练 recover transformer。

    训练主线可以概括为：
    1. 读取原始 latent 序列。
    2. 将其编码成低码率 low latents，模拟压缩传输结果。
    3. 构建“低码率历史 -> 高保真目标”的窗口数据集。
    4. 复用 Helios stage1 transformer 进行噪声预测训练。
    5. 保存权重、recover 配置和训练指标。
    """
    rank = int(os.environ.get("RANK", "0"))
    device = resolve_train_device(args.device)
    validate_train_args(args)

    # 多卡训练时给不同 rank 偏移 seed，既保证可复现，又避免完全相同的随机流。
    set_seed(args.seed + rank)
    weight_dtype = parse_weight_dtype(args.weight_dtype)
    codec_config = build_codec_config(args)
    history_sizes = normalize_history_sizes(args.history_sizes)
    input_root, latent_paths = discover_input_latent_paths(args.input_path)
    # 每个输入样本都会同时准备 clean / low 两套 latent 表示，为训练监督和条件输入服务。
    prepared_sequences = [prepare_sequence(path, codec_config) for path in latent_paths]
    dataset = LatentWindowDataset(
        sequences=prepared_sequences,
        history_sizes=history_sizes,
        latent_window_size=args.latent_window_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training windows were built from the provided latent files.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    transformer, _scheduler, prompt_embeds = load_transformer_bundle(
        base_model_path=args.base_model_path,
        device=device,
        weight_dtype=weight_dtype,
        checkpoint_dir=None,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    transformer.train()
    base_transformer = transformer
    trainable_params = count_trainable_parameters(base_transformer)

    # 只优化 recover transformer 中可训练的参数，不改动额外模块。
    optimizer = torch.optim.AdamW(
        [param for param in transformer.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    accelerator = create_train_accelerator(
        weight_dtype=args.weight_dtype,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if accelerator.state.deepspeed_plugin is not None:
        # 显式同步 micro batch 配置，避免 deepspeed 默认值与当前脚本参数不一致。
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    output_dir = args.output_dir.resolve()
    checkpoint_save_interval_epochs = max(1, math.ceil(args.epochs / 10))
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[train] input_root={input_root} files={len(latent_paths)} windows={len(dataset)} "
            f"device={device} world_size={accelerator.num_processes} "
            f"effective_global_batch_size={args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
            f"save_every_epochs={checkpoint_save_interval_epochs} "
            f"loss_weighting_scheme={args.loss_weighting_scheme} "
            f"flow/x/noise=({args.flow_loss_weight:.3f}/{args.x_loss_weight:.3f}/{args.noise_loss_weight:.3f}) "
            f"aux_warmup={args.aux_loss_warmup_steps} aux_ramp={args.aux_loss_ramp_steps}"
        )
    accelerator.wait_for_everyone()

    train_metrics: Dict[str, object] = {
        "checkpoint_format_version": RECOVER_CONFIG_VERSION,
        "losses": [],
        "steps": 0,
        "trainable_params": trainable_params,
        "world_size": accelerator.num_processes,
        "per_device_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "input_path": str(args.input_path.resolve()),
        "num_sequences": len(prepared_sequences),
        "loss_weighting_scheme": args.loss_weighting_scheme,
        "logit_mean": args.logit_mean,
        "logit_std": args.logit_std,
        "mode_scale": args.mode_scale,
        "loss_ema_decay": args.loss_ema_decay,
        "flow_loss_weight": args.flow_loss_weight,
        "x_loss_weight": args.x_loss_weight,
        "noise_loss_weight": args.noise_loss_weight,
        "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
        "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        "loss_definition": {
            "optimized_loss": (
                "flow_loss_weight * masked_weighted_mse(flow_pred, noise-latent) + "
                "aux_scale * (x_loss_weight * normalized_mse(x_pred, latent) + "
                "noise_loss_weight * normalized_mse(noise_pred, noise))"
            ),
            "raw_loss": "masked_mse(flow_pred, noise-latent)",
            "x_prediction": "x_pred = x_t - sigma * flow_pred",
            "noise_prediction": "noise_pred = x_t + (1 - sigma) * flow_pred",
        },
        "raw_losses": [],
        "losses_ema": [],
        "flow_losses": [],
        "x_losses": [],
        "noise_losses": [],
        "aux_scales": [],
        "sigma_means": [],
        "flow_target_rms": [],
        "flow_pred_rms": [],
        "x_target_rms": [],
        "x_pred_rms": [],
        "noise_target_rms": [],
        "noise_pred_rms": [],
        "epoch_summaries": [],
        "checkpoint_save_interval_epochs": checkpoint_save_interval_epochs,
        "saved_checkpoints": [],
    }

    # 训练日志既保留逐 step 记录，也保留逐 epoch 汇总，方便看短期波动和长期趋势。
    progress = None
    wandb_run = None
    try:
        if accelerator.is_main_process:
            wandb_run = init_offline_wandb_run(
                output_dir=output_dir,
                args=args,
                codec_config=codec_config,
                history_sizes=history_sizes,
                latent_window_size=args.latent_window_size,
                checkpoint_save_interval_epochs=checkpoint_save_interval_epochs,
            )

        if args.max_steps is not None and accelerator.is_main_process:
            progress = tqdm(total=args.max_steps, desc="Training")

        global_step = 0
        stop_training = False
        running_metrics: Dict[str, torch.Tensor] = {}
        running_micro_steps = 0
        loss_ema_value: Optional[float] = None
        for epoch in range(args.epochs):
            # epoch 内先累计，再在真正 sync gradients 后统一做 reduce 和日志记录，
            # 这样与梯度累积的语义保持一致。
            epoch_metric_sums = {
                "loss": 0.0,
                "raw_loss": 0.0,
                "flow_loss": 0.0,
                "x_loss": 0.0,
                "noise_loss": 0.0,
                "aux_scale": 0.0,
                "sigma_mean": 0.0,
                "flow_target_rms": 0.0,
                "flow_pred_rms": 0.0,
                "x_target_rms": 0.0,
                "x_pred_rms": 0.0,
                "noise_target_rms": 0.0,
                "noise_pred_rms": 0.0,
            }
            epoch_logged_steps = 0
            epoch_iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                disable=not accelerator.is_main_process,
            )
            for batch in epoch_iterator:
                with accelerator.accumulate(transformer):
                    step_output = training_step(
                        batch=batch,
                        transformer=transformer,
                        prompt_embeds=prompt_embeds,
                        device=device,
                        weight_dtype=weight_dtype,
                        history_sizes=history_sizes,
                        latent_window_size=args.latent_window_size,
                        loss_weighting_scheme=args.loss_weighting_scheme,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                        flow_loss_weight=args.flow_loss_weight,
                        x_loss_weight=args.x_loss_weight,
                        noise_loss_weight=args.noise_loss_weight,
                        aux_loss_warmup_steps=args.aux_loss_warmup_steps,
                        aux_loss_ramp_steps=args.aux_loss_ramp_steps,
                        global_step=global_step,
                    )
                    accelerator.backward(step_output.loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                detached_metrics = step_output.detached_float_metrics()
                if not running_metrics:
                    running_metrics = detached_metrics
                else:
                    for metric_name, metric_value in detached_metrics.items():
                        running_metrics[metric_name] = running_metrics[metric_name] + metric_value
                running_micro_steps += 1

                if not accelerator.sync_gradients:
                    continue

                # 一个全局优化步完成后，再把各 rank 的统计量聚合到主进程记录。
                global_step += 1
                reduced_metrics = {
                    metric_name: accelerator.reduce(metric_value / running_micro_steps, reduction="mean")
                    for metric_name, metric_value in running_metrics.items()
                }
                running_metrics = {}
                running_micro_steps = 0
                if accelerator.is_main_process:
                    scalar_metrics = {
                        metric_name: float(metric_value.cpu().item())
                        for metric_name, metric_value in reduced_metrics.items()
                    }
                    loss_value = scalar_metrics["loss"]
                    raw_loss_value = scalar_metrics["raw_loss"]
                    flow_loss_value = scalar_metrics["flow_loss"]
                    x_loss_value = scalar_metrics["x_loss"]
                    noise_loss_value = scalar_metrics["noise_loss"]
                    aux_scale_value = scalar_metrics["aux_scale"]
                    sigma_mean_value = scalar_metrics["sigma_mean"]
                    flow_target_rms_value = scalar_metrics["flow_target_rms"]
                    flow_pred_rms_value = scalar_metrics["flow_pred_rms"]
                    x_target_rms_value = scalar_metrics["x_target_rms"]
                    x_pred_rms_value = scalar_metrics["x_pred_rms"]
                    noise_target_rms_value = scalar_metrics["noise_target_rms"]
                    noise_pred_rms_value = scalar_metrics["noise_pred_rms"]
                    if loss_ema_value is None:
                        loss_ema_value = loss_value
                    else:
                        loss_ema_value = args.loss_ema_decay * loss_ema_value + (1.0 - args.loss_ema_decay) * loss_value

                    train_metrics["steps"] = global_step
                    train_metrics["losses"].append(loss_value)
                    train_metrics["raw_losses"].append(raw_loss_value)
                    train_metrics["losses_ema"].append(loss_ema_value)
                    train_metrics["flow_losses"].append(flow_loss_value)
                    train_metrics["x_losses"].append(x_loss_value)
                    train_metrics["noise_losses"].append(noise_loss_value)
                    train_metrics["aux_scales"].append(aux_scale_value)
                    train_metrics["sigma_means"].append(sigma_mean_value)
                    train_metrics["flow_target_rms"].append(flow_target_rms_value)
                    train_metrics["flow_pred_rms"].append(flow_pred_rms_value)
                    train_metrics["x_target_rms"].append(x_target_rms_value)
                    train_metrics["x_pred_rms"].append(x_pred_rms_value)
                    train_metrics["noise_target_rms"].append(noise_target_rms_value)
                    train_metrics["noise_pred_rms"].append(noise_pred_rms_value)

                    epoch_metric_sums["loss"] += loss_value
                    epoch_metric_sums["raw_loss"] += raw_loss_value
                    epoch_metric_sums["flow_loss"] += flow_loss_value
                    epoch_metric_sums["x_loss"] += x_loss_value
                    epoch_metric_sums["noise_loss"] += noise_loss_value
                    epoch_metric_sums["aux_scale"] += aux_scale_value
                    epoch_metric_sums["sigma_mean"] += sigma_mean_value
                    epoch_metric_sums["flow_target_rms"] += flow_target_rms_value
                    epoch_metric_sums["flow_pred_rms"] += flow_pred_rms_value
                    epoch_metric_sums["x_target_rms"] += x_target_rms_value
                    epoch_metric_sums["x_pred_rms"] += x_pred_rms_value
                    epoch_metric_sums["noise_target_rms"] += noise_target_rms_value
                    epoch_metric_sums["noise_pred_rms"] += noise_pred_rms_value
                    epoch_logged_steps += 1

                    if global_step % args.log_every == 0:
                        epoch_iterator.set_postfix(
                            loss=f"{loss_value:.6f}",
                            ema=f"{loss_ema_value:.6f}",
                            flow=f"{flow_loss_value:.6f}",
                            x=f"{x_loss_value:.6f}",
                            noise=f"{noise_loss_value:.6f}",
                            sigma=f"{sigma_mean_value:.3f}",
                        )
                    if progress is not None:
                        progress.update(1)
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": loss_value,
                                "train/raw_loss": raw_loss_value,
                                "train/flow_loss": flow_loss_value,
                                "train/x_loss": x_loss_value,
                                "train/noise_loss": noise_loss_value,
                                "train/aux_scale": aux_scale_value,
                                "train/loss_ema": loss_ema_value,
                                "train/sigma_mean": sigma_mean_value,
                                "train/flow_target_rms": flow_target_rms_value,
                                "train/flow_pred_rms": flow_pred_rms_value,
                                "train/x_target_rms": x_target_rms_value,
                                "train/x_pred_rms": x_pred_rms_value,
                                "train/noise_target_rms": noise_target_rms_value,
                                "train/noise_pred_rms": noise_pred_rms_value,
                                "train/epoch": epoch + 1,
                            },
                            step=global_step,
                        )

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

            # 周期性 checkpoint 方便观察不同训练阶段的恢复质量，也利于中断后续训。
            should_save_periodic_checkpoint = global_step > 0 and (epoch + 1) % checkpoint_save_interval_epochs == 0
            if accelerator.is_main_process and epoch_logged_steps > 0:
                epoch_summary = {
                    "epoch": epoch + 1,
                    "steps": epoch_logged_steps,
                    "loss_mean": epoch_metric_sums["loss"] / epoch_logged_steps,
                    "raw_loss_mean": epoch_metric_sums["raw_loss"] / epoch_logged_steps,
                    "flow_loss_mean": epoch_metric_sums["flow_loss"] / epoch_logged_steps,
                    "x_loss_mean": epoch_metric_sums["x_loss"] / epoch_logged_steps,
                    "noise_loss_mean": epoch_metric_sums["noise_loss"] / epoch_logged_steps,
                    "aux_scale_mean": epoch_metric_sums["aux_scale"] / epoch_logged_steps,
                    "loss_ema_last": loss_ema_value,
                    "sigma_mean": epoch_metric_sums["sigma_mean"] / epoch_logged_steps,
                    "flow_target_rms_mean": epoch_metric_sums["flow_target_rms"] / epoch_logged_steps,
                    "flow_pred_rms_mean": epoch_metric_sums["flow_pred_rms"] / epoch_logged_steps,
                    "x_target_rms_mean": epoch_metric_sums["x_target_rms"] / epoch_logged_steps,
                    "x_pred_rms_mean": epoch_metric_sums["x_pred_rms"] / epoch_logged_steps,
                    "noise_target_rms_mean": epoch_metric_sums["noise_target_rms"] / epoch_logged_steps,
                    "noise_pred_rms_mean": epoch_metric_sums["noise_pred_rms"] / epoch_logged_steps,
                }
                train_metrics["epoch_summaries"].append(epoch_summary)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch/loss_mean": epoch_summary["loss_mean"],
                            "epoch/raw_loss_mean": epoch_summary["raw_loss_mean"],
                            "epoch/flow_loss_mean": epoch_summary["flow_loss_mean"],
                            "epoch/x_loss_mean": epoch_summary["x_loss_mean"],
                            "epoch/noise_loss_mean": epoch_summary["noise_loss_mean"],
                            "epoch/aux_scale_mean": epoch_summary["aux_scale_mean"],
                            "epoch/loss_ema_last": epoch_summary["loss_ema_last"],
                            "epoch/sigma_mean": epoch_summary["sigma_mean"],
                            "epoch/flow_target_rms_mean": epoch_summary["flow_target_rms_mean"],
                            "epoch/flow_pred_rms_mean": epoch_summary["flow_pred_rms_mean"],
                            "epoch/x_target_rms_mean": epoch_summary["x_target_rms_mean"],
                            "epoch/x_pred_rms_mean": epoch_summary["x_pred_rms_mean"],
                            "epoch/noise_target_rms_mean": epoch_summary["noise_target_rms_mean"],
                            "epoch/noise_pred_rms_mean": epoch_summary["noise_pred_rms_mean"],
                        },
                        step=global_step,
                    )
            if should_save_periodic_checkpoint:
                periodic_checkpoint_dir = build_periodic_checkpoint_dir(
                    output_dir=output_dir,
                    epoch=epoch + 1,
                    global_step=global_step,
                )
                if accelerator.is_main_process:
                    checkpoint_record = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "checkpoint_dir": str(periodic_checkpoint_dir),
                    }
                    train_metrics["saved_checkpoints"].append(checkpoint_record)
                    print(
                        f"[train] saving periodic checkpoint epoch={epoch + 1} "
                        f"step={global_step} dir={periodic_checkpoint_dir}"
                    )
                save_training_artifacts(
                    output_dir=periodic_checkpoint_dir,
                    transformer=transformer,
                    codec_config=codec_config,
                    history_sizes=history_sizes,
                    latent_window_size=args.latent_window_size,
                    args=args,
                    train_metrics=train_metrics,
                    accelerator=accelerator,
                )
                if accelerator.is_main_process and wandb_run is not None:
                    wandb_run.log(
                        {
                            "checkpoint/epoch": epoch + 1,
                            "checkpoint/global_step": global_step,
                        },
                        step=global_step,
                    )

            if stop_training:
                break

        # 训练结束后无论是否提前停止，都保存最终产物，作为默认推理入口 checkpoint。
        save_training_artifacts(
            output_dir=output_dir,
            transformer=transformer,
            codec_config=codec_config,
            history_sizes=history_sizes,
            latent_window_size=args.latent_window_size,
            args=args,
            train_metrics=train_metrics,
            accelerator=accelerator,
        )
        if accelerator.is_main_process and train_metrics["losses"]:
            if wandb_run is not None:
                wandb_run.summary["train/steps"] = train_metrics["steps"]
                wandb_run.summary["train/last_loss"] = train_metrics["losses"][-1]
                wandb_run.summary["train/last_raw_loss"] = train_metrics["raw_losses"][-1]
                wandb_run.summary["train/last_flow_loss"] = train_metrics["flow_losses"][-1]
                wandb_run.summary["train/last_x_loss"] = train_metrics["x_losses"][-1]
                wandb_run.summary["train/last_noise_loss"] = train_metrics["noise_losses"][-1]
                wandb_run.summary["train/last_loss_ema"] = train_metrics["losses_ema"][-1]
                wandb_run.summary["train/trainable_params"] = train_metrics["trainable_params"]
            print(
                f"[train] completed steps={train_metrics['steps']} "
                f"last_loss={train_metrics['losses'][-1]:.6f} "
                f"last_loss_ema={train_metrics['losses_ema'][-1]:.6f} "
                f"last_flow_loss={train_metrics['flow_losses'][-1]:.6f} "
                f"last_x_loss={train_metrics['x_losses'][-1]:.6f} "
                f"last_noise_loss={train_metrics['noise_losses'][-1]:.6f} "
                f"last_raw_loss={train_metrics['raw_losses'][-1]:.6f} "
                f"trainable_params={train_metrics['trainable_params']}"
            )
    finally:
        if progress is not None:
            progress.close()
        if accelerator.is_main_process and wandb_run is not None:
            # 使用 finally 确保异常退出时也能正常结束 wandb run，避免日志目录损坏。
            wandb_run.finish()


def command_infer(args: argparse.Namespace) -> None:
    """执行压缩与恢复推理。

    推理链路刻意保持贴近业务主线：
    1. 读取训练阶段保存的 recover 配置和权重。
    2. 对输入 clean latents 做低码率编码，得到 low_latents。
    3. 用 recover transformer 恢复出 recover_latents。
    4. 保存 low_latents、recover_latents 与恢复指标。

    默认不直接生成视频，避免偏离“压缩-传输-恢复”验证主线。
    """
    distributed_context, device = init_distributed_context(args.device)
    try:
        set_seed(args.seed)
        checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint_config = load_recover_config(checkpoint_dir)
        # 推理阶段优先沿用 checkpoint 的关键配置，除非用户在 CLI 中显式覆盖。
        base_model_path = resolve_infer_value(
            cli_value=args.base_model_path,
            default_value=DEFAULT_BASE_MODEL_PATH,
            checkpoint_value=checkpoint_config.get("base_model_path"),
        )
        weight_dtype_name = resolve_infer_value(
            cli_value=args.weight_dtype,
            default_value=DEFAULT_WEIGHT_DTYPE,
            checkpoint_value=checkpoint_config.get("weight_dtype"),
        )
        codec_config = CodecConfig(**checkpoint_config["codec_config"])
        history_sizes = normalize_history_sizes(checkpoint_config["history_sizes"])
        latent_window_size = int(checkpoint_config["latent_window_size"])
        weight_dtype = parse_weight_dtype(str(weight_dtype_name))

        input_root, latent_paths = discover_input_latent_paths(args.input_path)
        output_dir = args.output_dir.resolve()
        low_dir = output_dir / "low_latents"
        recover_dir = output_dir / "recover_latents"
        metrics_dir = output_dir / "metrics"

        if distributed_context.is_main_process:
            low_dir.mkdir(parents=True, exist_ok=True)
            recover_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[infer] input_root={input_root} files={len(latent_paths)} "
                f"device={device} world_size={distributed_context.world_size}"
            )
        distributed_barrier(distributed_context)

        transformer, scheduler, prompt_embeds = load_transformer_bundle(
            base_model_path=str(base_model_path),
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_dir=checkpoint_dir,
            gradient_checkpointing=False,
        )
        transformer.eval()

        for sample_idx, latent_path in enumerate(latent_paths, start=1):
            if distributed_context.is_main_process:
                print(f"[infer] ({sample_idx}/{len(latent_paths)}) input={latent_path}")

            sequence = prepare_sequence(latent_path, codec_config)
            recovered_full_latents = reconstruct_sequence(
                sequence=sequence,
                transformer=transformer,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                device=device,
                weight_dtype=weight_dtype,
                history_sizes=history_sizes,
                latent_window_size=latent_window_size,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                distributed_context=distributed_context,
            )

            if distributed_context.is_main_process:
                if recovered_full_latents is None:
                    raise RuntimeError("Main process did not receive reconstructed latent windows.")

                low_latent_path, recover_latent_path, metrics_path = resolve_output_paths(
                    input_path=latent_path,
                    input_root=input_root,
                    low_dir=low_dir,
                    recover_dir=recover_dir,
                    metrics_dir=metrics_dir,
                )
                low_payload = build_low_latent_payload(sequence)
                recover_payload = build_recover_latent_payload(
                    sequence=sequence,
                    recovered_full_latents=recovered_full_latents,
                    base_model_path=str(base_model_path),
                    checkpoint_dir=checkpoint_dir,
                )
                torch.save(low_payload, low_latent_path)
                torch.save(recover_payload, recover_latent_path)

                # direct_low_metrics 衡量“只做低码率编解码、不做 recover”时的基线失真；
                # restored_metrics 衡量 recover 模块真正带来的修复收益。
                direct_low_metrics = compute_tensor_metrics(sequence.low_full_latents, sequence.clean_full_latents)
                restored_metrics = compute_tensor_metrics(recovered_full_latents, sequence.clean_full_latents)
                boundary_l1 = compute_boundary_transition_l1(
                    prediction=recovered_full_latents,
                    target=sequence.clean_full_latents,
                    latent_window_size=latent_window_size,
                )
                metrics_payload = {
                    "input_path": str(sequence.path),
                    "low_latent_path": str(low_latent_path),
                    "recover_latent_path": str(recover_latent_path),
                    "raw_file_bytes": sequence.metadata.raw_file_bytes,
                    "raw_bpp": sequence.metadata.raw_bpp,
                    "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
                    "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
                    "direct_low_metrics": direct_low_metrics,
                    "restored_metrics": restored_metrics,
                    "boundary_transition_l1": boundary_l1,
                    "codec_config": asdict(codec_config),
                    "checkpoint_dir": str(checkpoint_dir),
                    "num_inference_steps": args.num_inference_steps,
                }
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(metrics_payload, handle, indent=2)

                print(
                    f"[infer] saved low={low_latent_path} recover={recover_latent_path} "
                    f"low_bpp={sequence.low_codec_payload['low_bpp']:.6f} "
                    f"direct_low_l1={direct_low_metrics['l1']:.6f} restored_l1={restored_metrics['l1']:.6f}"
                )

            distributed_barrier(distributed_context)
            if device.type == "cuda":
                # 逐样本清 cache，减轻长序列推理时的显存峰值压力。
                torch.cuda.empty_cache()
    finally:
        cleanup_distributed_context(distributed_context)


def build_codec_config(args: argparse.Namespace) -> CodecConfig:
    """从命令行参数构造低码率编码配置。"""
    if args.temporal_factor < 1 or args.spatial_factor < 1:
        raise ValueError("temporal_factor and spatial_factor must both be >= 1.")
    return CodecConfig(
        temporal_factor=args.temporal_factor,
        spatial_factor=args.spatial_factor,
        quant_dtype=args.quant_dtype,
        keyframe_dtype=args.keyframe_dtype,
    )


def normalize_history_sizes(history_sizes: Sequence[int]) -> List[int]:
    """把 history_sizes 规范化为从大到小的列表。

    Helios 的 stage1 输入里会区分 long / mid / short 三档历史，
    这里统一排序，保证后续传参顺序稳定。
    """
    normalized = sorted([int(value) for value in history_sizes], reverse=True)
    if sum(normalized) <= 0:
        raise ValueError("history_sizes must sum to a positive value.")
    return normalized


def discover_input_latent_paths(input_path: Path) -> Tuple[Path, List[Path]]:
    """发现待处理的 latent 文件。

    支持两种输入形态：
    - 单个 `.pt` 文件
    - 包含多个 `.pt` 文件的目录
    """
    resolved = input_path.resolve()
    if resolved.is_file():
        if resolved.suffix != ".pt":
            raise ValueError(f"Expected a .pt latent file, got: {resolved}")
        return resolved.parent, [resolved]

    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Input path must be a latent file or directory: {resolved}")

    latent_paths = sorted(path for path in resolved.rglob("*.pt") if path.is_file())
    if not latent_paths:
        raise FileNotFoundError(f"No latent files found in: {resolved}")
    return resolved, latent_paths


def prepare_sequence(path: Path, codec_config: CodecConfig) -> PreparedSequence:
    """把单个 latent 文件预处理成 recover 主链路需要的统一结构。

    这里会同时算出：
    - clean_full_latents: 原始高保真表示
    - low_codec_payload: 低码率编码结果
    - low_full_latents: 低码率解码后的粗恢复结果

    这样训练与推理都能围绕同一份中间表示展开。
    """
    payload = load_payload(path)
    validate_payload(payload, path)
    clean_full_latents = flatten_latent_chunks(payload["latent_chunks"]).float().contiguous()
    chunk_lengths = [int(chunk.shape[1]) for chunk in payload["latent_chunks"]]
    chunk_frame_ranges = build_chunk_frame_ranges(
        [int((chunk.shape[1] - 1) * 4 + 1) for chunk in payload["latent_chunks"]]
    )
    source_num_frames = infer_source_num_frames(payload["latent_chunks"], payload)
    source_width, source_height = resolve_source_resolution(payload, path)
    total_pixels = source_num_frames * source_width * source_height
    raw_file_bytes = path.stat().st_size
    raw_bpp = compute_bpp_from_total_pixels(raw_file_bytes, total_pixels)
    # 先编码，再立即解码出 low_full_latents，模拟接收端只拿到低码率传输结果时的状态。
    low_codec_payload = encode_low_latents(clean_full_latents, codec_config, chunk_lengths)
    low_codec_payload["low_codec_bytes"] = estimate_low_codec_bytes(low_codec_payload)
    low_codec_payload["low_bpp"] = compute_bpp_from_total_pixels(
        int(low_codec_payload["low_codec_bytes"]),
        total_pixels,
    )
    low_full_latents = decode_low_latents(low_codec_payload).float().contiguous()
    metadata = SequenceMetadata(
        source_fps=float(payload["source_fps"]),
        source_num_frames=source_num_frames,
        source_width=source_width,
        source_height=source_height,
        total_pixels=total_pixels,
        raw_file_bytes=raw_file_bytes,
        raw_bpp=raw_bpp,
        chunk_lengths=chunk_lengths,
        chunk_frame_ranges=chunk_frame_ranges,
    )
    return PreparedSequence(
        path=path.resolve(),
        metadata=metadata,
        clean_full_latents=clean_full_latents,
        low_codec_payload=low_codec_payload,
        low_full_latents=low_full_latents,
    )


def encode_low_latents(
    clean_full_latents: torch.Tensor,
    codec_config: CodecConfig,
    chunk_lengths: Sequence[int],
) -> Dict[str, object]:
    """把 clean latents 编码成低码率载荷。

    当前策略强调“最小可用”的低码率链路验证：
    - 首帧单独保留高精度，减少序列起点漂移。
    - 其余帧时空下采样，进一步降低冗余。
    - 下采样结果按通道做对称量化，得到可传输的低比特表示。
    """
    if clean_full_latents.ndim != 4:
        raise ValueError(f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}")

    keyframe_dtype = torch.float16 if codec_config.keyframe_dtype == "float16" else torch.float32
    # 第一帧直接当作关键帧保留，后续恢复都围绕它展开。
    keyframe = clean_full_latents[:, :1].to(keyframe_dtype).contiguous()
    remainder = clean_full_latents[:, 1:]

    if remainder.shape[1] == 0:
        return {
            "codec_config": asdict(codec_config),
            "chunk_lengths": [int(length) for length in chunk_lengths],
            "keyframe": keyframe.cpu(),
            "quantized_remainder": torch.empty(0, dtype=torch.int8),
            "scales": torch.empty(0, dtype=torch.float32),
            "reduced_shape": [int(clean_full_latents.shape[0]), 0, int(clean_full_latents.shape[2]), int(clean_full_latents.shape[3])],
            "original_remainder_shape": [int(value) for value in remainder.shape],
        }

    # 三线性插值统一处理时间和空间降采样，生成低码率 remainder。
    reduced_remainder = trilinear_resize(
        remainder.unsqueeze(0),
        (
            max(1, math.ceil(remainder.shape[1] / codec_config.temporal_factor)),
            max(1, math.ceil(remainder.shape[2] / codec_config.spatial_factor)),
            max(1, math.ceil(remainder.shape[3] / codec_config.spatial_factor)),
        ),
    ).squeeze(0)
    quantized_remainder, scales = symmetric_quantize_per_channel(reduced_remainder, codec_config.quant_dtype)
    return {
        "codec_config": asdict(codec_config),
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "keyframe": keyframe.cpu(),
        "quantized_remainder": quantized_remainder.cpu(),
        "scales": scales.cpu(),
        "reduced_shape": [int(value) for value in reduced_remainder.shape],
        "original_remainder_shape": [int(value) for value in remainder.shape],
    }


def decode_low_latents(codec_payload: Dict[str, object]) -> torch.Tensor:
    """从低码率载荷还原出粗恢复 latent。

    这个结果不是最终的 recover_latents，而是：
    - 训练时的历史条件输入
    - 推理时接收端可直接得到的低质量基线
    """
    keyframe = codec_payload["keyframe"].float()
    quantized_remainder = codec_payload["quantized_remainder"]
    original_remainder_shape = tuple(int(value) for value in codec_payload["original_remainder_shape"])
    if isinstance(quantized_remainder, torch.Tensor) and quantized_remainder.numel() == 0:
        return keyframe.float()

    reduced_remainder = symmetric_dequantize_per_channel(
        quantized_remainder=codec_payload["quantized_remainder"],
        scales=codec_payload["scales"],
    )
    # 解量化后按原始 remainder 尺寸上采样，恢复成与 clean latent 同 shape 的粗结果。
    restored_remainder = trilinear_resize(
        reduced_remainder.unsqueeze(0),
        (original_remainder_shape[1], original_remainder_shape[2], original_remainder_shape[3]),
    ).squeeze(0)
    return torch.cat([keyframe.float(), restored_remainder.float()], dim=1)


def symmetric_quantize_per_channel(latents: torch.Tensor, quant_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """按通道做对称 int8 量化，并记录每个通道的 scale。"""
    if quant_dtype != "int8":
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}. Only int8 is implemented.")
    scales = latents.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(EPS) / 127.0
    quantized = torch.round(latents / scales).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.squeeze(-1).squeeze(-1).squeeze(-1).float().contiguous()


def symmetric_dequantize_per_channel(quantized_remainder: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """按通道 scale 把量化 remainder 还原回浮点数。"""
    if quantized_remainder.ndim != 4:
        raise ValueError(
            f"Expected quantized remainder with shape [C, T, H, W], got {tuple(quantized_remainder.shape)}"
        )
    scale_view = scales.view(-1, 1, 1, 1).to(dtype=torch.float32)
    return quantized_remainder.float() * scale_view


def trilinear_resize(latents: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """统一封装 3D latent 的时空插值。"""
    return F.interpolate(latents.float(), size=size, mode="trilinear", align_corners=False)


def estimate_low_codec_bytes(codec_payload: Dict[str, object]) -> int:
    """估算低码率载荷的字节数。

    这里统计的是 tensor 数据本身加最小必要 shape 元数据，
    用来近似衡量这条低码率链路的传输成本。
    """
    total_bytes = 0
    for key in ("keyframe", "quantized_remainder", "scales"):
        value = codec_payload[key]
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()

    # Count only the minimal integer shape metadata needed to decode the payload.
    metadata_values: List[int] = [len(codec_payload["chunk_lengths"])]
    metadata_values.extend(int(value) for value in codec_payload["chunk_lengths"])
    metadata_values.extend(int(value) for value in codec_payload["reduced_shape"])
    metadata_values.extend(int(value) for value in codec_payload["original_remainder_shape"])
    total_bytes += len(metadata_values) * 4
    return int(total_bytes)


def extract_target_window(
    clean_full_latents: torch.Tensor,
    section_start: int,
    latent_window_size: int,
) -> Tuple[torch.Tensor, int]:
    """从 clean latents 中截取一个目标恢复窗口。

    如果最后一个窗口长度不足 `latent_window_size`，则用最后一帧重复补齐；
    同时返回 `valid_target_frames`，后续通过 mask 避免 padding 帧影响 loss。
    """
    target = clean_full_latents[:, section_start : section_start + latent_window_size]
    valid_target_frames = target.shape[1]
    if valid_target_frames == 0:
        raise ValueError(f"Section start {section_start} produced an empty target window.")
    if valid_target_frames < latent_window_size:
        padding = target[:, -1:].repeat(1, latent_window_size - valid_target_frames, 1, 1)
        target = torch.cat([target, padding], dim=1)
    return target.contiguous(), valid_target_frames


def extract_history_window(
    low_full_latents: torch.Tensor,
    section_start: int,
    history_window_size: int,
) -> torch.Tensor:
    """从 low latents 中截取历史窗口。

    注意这里故意只取 `low_full_latents[:, 1:]`：
    - 首帧已单独作为 `x0_latents` 输入
    - 历史窗口只负责提供后续低码率上下文
    """
    low_remainder = low_full_latents[:, 1:]
    remainder_start = max(0, (section_start - 1) - history_window_size)
    remainder_end = max(0, section_start - 1)
    history = low_remainder[:, remainder_start:remainder_end]
    if history.shape[1] < history_window_size:
        # 序列开头历史不足时左侧补零，保持输入 shape 稳定。
        zeros = torch.zeros(
            history.shape[0],
            history_window_size - history.shape[1],
            history.shape[2],
            history.shape[3],
            dtype=history.dtype,
        )
        history = torch.cat([zeros, history], dim=1)
    return history.contiguous()


def load_transformer_bundle(
    base_model_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    checkpoint_dir: Optional[Path],
    gradient_checkpointing: bool,
    accelerator=None,
) -> Tuple[torch.nn.Module, object, torch.Tensor]:
    """加载 recover 所需的 Helios transformer、scheduler 和空 prompt embedding。

    这里复用 Helios 原始 transformer 结构：
    - 训练时从 base model 初始化并开放梯度
    - 推理时从 checkpoint 恢复权重并冻结

    文本编码器只用于构造一份共享的空 prompt embedding，之后即可释放。
    """
    HeliosTransformer3DModel, HeliosScheduler, encode_prompt = import_helios_inference_stack()
    zero3_disabled = nullcontext()
    if accelerator is not None and accelerator.state.deepspeed_plugin is not None:
        # The prompt text encoder is only used to produce a shared empty-prompt embedding and
        # the transformer is later sharded by `accelerator.prepare(...)`, so disable ZeRO-3 init
        # during `from_pretrained` to keep Hugging Face model loading behavior intact.
        zero3_disabled = accelerator.state.deepspeed_plugin.zero3_init_context_manager(enable=False)

    with zero3_disabled:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
        ).to(device)
        with torch.inference_mode():
            prompt_embeds, _ = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt="",
                device=device,
                dtype=weight_dtype,
            )
        del tokenizer
        text_encoder.to("cpu")
        del text_encoder

        transformer = HeliosTransformer3DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
    transformer.to(device)
    scheduler = HeliosScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    if checkpoint_dir is None:
        # 训练模式：从 base transformer 初始化，后续参数参与优化。
        transformer.requires_grad_(True)
    else:
        state_dict_path = checkpoint_dir / "transformer_full.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Missing transformer checkpoint: {state_dict_path}")
        state_dict = load_state_dict_file(state_dict_path)
        incompatible = transformer.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected state dict mismatch while loading {state_dict_path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        # 推理模式：只做前向恢复，不需要梯度。
        transformer.requires_grad_(False)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    return transformer, scheduler, prompt_embeds.to(device=device, dtype=weight_dtype)


def training_step(
    batch: Dict[str, torch.Tensor | int],
    transformer: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    history_sizes: Sequence[int],
    latent_window_size: int,
    loss_weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    flow_loss_weight: float,
    x_loss_weight: float,
    noise_loss_weight: float,
    aux_loss_warmup_steps: int,
    aux_loss_ramp_steps: int,
    global_step: int,
) -> TrainingStepOutput:
    """执行一次 recover 训练步。

    训练目标延续 Helios stage1 的 flow 预测范式：
    - 先把目标 latent 按 sigma 混入噪声
    - 让 transformer 预测 `flow = noise - latent`
    - 再由同一个 `flow_pred` 解析出 `x_pred` 和 `noise_pred`
    - 组合成 `flow + x + noise` 多目标监督

    其中历史条件来自 low latents，因此模型学到的是“如何从低码率上下文恢复高保真细节”。
    """
    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3 = import_diffusers_training_utils()
    history_latents = batch["history_latents"].to(device=device, dtype=weight_dtype)
    target_latents = batch["target_latents"].to(device=device, dtype=weight_dtype)
    x0_latents = batch["x0_latents"].to(device=device, dtype=weight_dtype)
    valid_target_frames = batch["valid_target_frames"].to(device=device)

    (
        model_input,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    ) = prepare_stage1_clean_input_from_latents(
        history_latents=history_latents,
        target_latents=target_latents,
        x0_latents=x0_latents,
        latent_window_size=latent_window_size,
        history_sizes=list(history_sizes),
        is_random_drop=False,
        random_drop_v2v_ratio=0.0,
        random_drop_t2v_ratio=0.0,
        is_keep_x0=True,
        dtype=weight_dtype,
        device=device,
    )

    # 在 latent 空间显式加噪，训练模型适应不同噪声强度下的恢复任务。
    noise = torch.randn(model_input.shape, device=device, dtype=weight_dtype)
    sigma = compute_density_for_timestep_sampling(
        weighting_scheme=loss_weighting_scheme,
        batch_size=model_input.shape[0],
        logit_mean=logit_mean,
        logit_std=logit_std,
        mode_scale=mode_scale,
    ).to(device=device, dtype=torch.float32)
    sigma = sigma.clamp_(0.0, 0.999)
    sigma_view = sigma.to(dtype=weight_dtype).view(-1, 1, 1, 1, 1)
    noisy_model_input = (1.0 - sigma_view) * model_input + sigma_view * noise
    flow_target = noise - model_input
    timesteps = sigma * 1000.0

    base_transformer = unwrap_model(transformer)
    prompt_batch = prompt_embeds.repeat(model_input.shape[0], 1, 1)
    with base_transformer.cache_context("cond"):
        flow_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_batch,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )[0]

    # 序列尾部可能被 padding，因此 loss 只统计真实存在的 target 帧。
    mask = build_valid_mask(
        valid_target_frames=valid_target_frames,
        latent_window_size=latent_window_size,
        device=device,
        dtype=torch.float32,
    )
    sigma_view_float = sigma.view(-1, 1, 1, 1, 1)
    xt = noisy_model_input.float()
    x_target = model_input.float()
    noise_target = noise.float()
    flow_target = flow_target.float()
    flow_pred = flow_pred.float()
    x_pred = xt - sigma_view_float * flow_pred
    noise_pred = xt + (1.0 - sigma_view_float) * flow_pred

    flow_sq_error = (flow_pred - flow_target).pow(2)
    weighting = compute_loss_weighting_for_sd3(weighting_scheme=loss_weighting_scheme, sigmas=sigma)
    weighting = weighting.to(device=device, dtype=torch.float32)
    while weighting.ndim < flow_sq_error.ndim:
        weighting = weighting.unsqueeze(-1)

    aux_scale = compute_aux_loss_scale(
        global_step=global_step,
        warmup_steps=aux_loss_warmup_steps,
        ramp_steps=aux_loss_ramp_steps,
        device=device,
    )
    raw_loss = masked_mean(flow_sq_error, mask)
    flow_loss = masked_mean(flow_sq_error * weighting, mask)
    x_loss = normalized_masked_mse(x_pred, x_target, mask)
    noise_loss = normalized_masked_mse(noise_pred, noise_target, mask)
    loss = (
        flow_loss_weight * flow_loss
        + aux_scale * (x_loss_weight * x_loss + noise_loss_weight * noise_loss)
    )
    return TrainingStepOutput(
        loss=loss,
        raw_loss=raw_loss,
        flow_loss=flow_loss,
        x_loss=x_loss,
        noise_loss=noise_loss,
        aux_scale=aux_scale,
        sigma_mean=sigma.mean(),
        flow_target_rms=masked_mean(flow_target.pow(2), mask).sqrt(),
        flow_pred_rms=masked_mean(flow_pred.pow(2), mask).sqrt(),
        x_target_rms=masked_mean(x_target.pow(2), mask).sqrt(),
        x_pred_rms=masked_mean(x_pred.pow(2), mask).sqrt(),
        noise_target_rms=masked_mean(noise_target.pow(2), mask).sqrt(),
        noise_pred_rms=masked_mean(noise_pred.pow(2), mask).sqrt(),
    )


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """在 mask 指定的有效区域上计算均值。"""
    expanded_mask = mask.expand_as(values)
    return (values * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def normalized_masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """计算按目标能量归一化后的 masked MSE。"""
    numerator = masked_mean((prediction - target).pow(2), mask)
    denominator = masked_mean(target.pow(2), mask) + EPS
    return numerator / denominator


def compute_aux_loss_scale(
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """按 warmup + ramp 日程控制辅助损失权重。"""
    if global_step < warmup_steps:
        scale = 0.0
    elif ramp_steps == 0:
        scale = 1.0
    else:
        scale = min(1.0, float(global_step - warmup_steps) / float(ramp_steps))
    return torch.tensor(scale, device=device, dtype=torch.float32)


def build_valid_mask(
    valid_target_frames: torch.Tensor,
    latent_window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """为每个样本构造目标窗口的有效帧掩码。"""
    batch_size = valid_target_frames.shape[0]
    mask = torch.zeros(batch_size, 1, latent_window_size, 1, 1, device=device, dtype=dtype)
    for idx, valid in enumerate(valid_target_frames.tolist()):
        mask[idx, :, : int(valid)] = 1
    return mask


@torch.inference_mode()
def reconstruct_sequence(
    sequence: PreparedSequence,
    transformer: torch.nn.Module,
    scheduler: object,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    history_sizes: Sequence[int],
    latent_window_size: int,
    num_inference_steps: int,
    seed: int,
    distributed_context: DistributedContext,
) -> Optional[torch.Tensor]:
    """把一段 low latents 重建成 recover latents。

    这里按 section 分段恢复，每个 section 只依赖：
    - 首帧 `x0`
    - 低码率历史窗口

    这与训练时的数据组织保持一致，也更贴近真实压缩恢复链路中的分段重建过程。
    """
    clean_full = sequence.clean_full_latents
    low_full = sequence.low_full_latents
    if clean_full.shape[1] == 1:
        # 只有首帧时，不存在 remainder，直接返回原始结果即可。
        if distributed_context.is_main_process:
            return clean_full.clone().cpu().contiguous()
        return None

    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    history_sizes = list(history_sizes)
    history_window_size = sum(history_sizes)
    # 多卡推理时采用轮转切分 section，尽量让不同 rank 负载更均匀。
    section_starts = list(range(1, clean_full.shape[1], latent_window_size))
    assigned_section_starts = section_starts[distributed_context.rank :: distributed_context.world_size]
    local_section_results: List[Tuple[int, torch.Tensor]] = []
    dummy_target = torch.zeros(
        1,
        clean_full.shape[0],
        latent_window_size,
        clean_full.shape[2],
        clean_full.shape[3],
        device=device,
        dtype=weight_dtype,
    )
    x0_latents = clean_full[:, :1].unsqueeze(0).to(device=device, dtype=weight_dtype)

    for section_start in tqdm(
        assigned_section_starts,
        desc="Reconstructing local sections",
        disable=not distributed_context.is_main_process,
    ):
        history_latents = extract_history_window(
            low_full_latents=low_full,
            section_start=section_start,
            history_window_size=history_window_size,
        ).unsqueeze(0)
        valid_target_frames = min(latent_window_size, clean_full.shape[1] - section_start)

        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=history_latents.to(device=device, dtype=weight_dtype),
            target_latents=dummy_target,
            x0_latents=x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=False,
            random_drop_v2v_ratio=0.0,
            random_drop_t2v_ratio=0.0,
            is_keep_x0=True,
            dtype=weight_dtype,
            device=device,
        )

        # 当前窗口从随机噪声出发，经 scheduler 逐步去噪恢复成目标 latent。
        section_latents = run_stage1_denoise(
            transformer=transformer,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            device=device,
            weight_dtype=weight_dtype,
            latent_shape=(1, clean_full.shape[0], latent_window_size, clean_full.shape[2], clean_full.shape[3]),
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            num_inference_steps=num_inference_steps,
            seed=seed + section_start,
        )
        local_section_results.append((section_start, section_latents[0, :, :valid_target_frames].contiguous()))

    gathered_sections = gather_reconstructed_sections(local_section_results, distributed_context)
    if gathered_sections is None:
        return None

    # 主进程按起始位置重新组织各个 section，避免分布式收集后时序打乱。
    section_map: Dict[int, torch.Tensor] = {}
    for section_start, recovered_latents in gathered_sections:
        if section_start in section_map:
            raise RuntimeError(f"Duplicate reconstructed section received for section_start={section_start}.")
        section_map[section_start] = recovered_latents

    missing_sections = [section_start for section_start in section_starts if section_start not in section_map]
    if missing_sections:
        raise RuntimeError(f"Missing reconstructed sections on rank 0: {missing_sections}.")

    recovered_sections = [section_map[section_start] for section_start in section_starts]
    recovered_remainder = torch.cat(recovered_sections, dim=1)
    recovered_remainder = recovered_remainder[:, : clean_full.shape[1] - 1]
    # 首帧继续沿用 clean keyframe，最大限度减少起始锚点误差。
    return torch.cat([clean_full[:, :1].cpu(), recovered_remainder], dim=1).contiguous()


def gather_reconstructed_sections(
    local_section_results: List[Tuple[int, torch.Tensor]],
    distributed_context: DistributedContext,
) -> Optional[List[Tuple[int, torch.Tensor]]]:
    """把不同 rank 的 section 恢复结果收集到主进程。"""
    if not distributed_context.is_distributed:
        return local_section_results

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed reconstruction expected an initialized process group.")

    gathered_section_results: Optional[List[List[Tuple[int, torch.Tensor]]]] = None
    if distributed_context.is_main_process:
        gathered_section_results = [list() for _ in range(distributed_context.world_size)]

    dist.gather_object(
        local_section_results,
        object_gather_list=gathered_section_results,
        dst=0,
    )
    if not distributed_context.is_main_process:
        return None
    assert gathered_section_results is not None

    merged_section_results: List[Tuple[int, torch.Tensor]] = []
    for rank_sections in gathered_section_results:
        merged_section_results.extend(rank_sections)
    return merged_section_results


@torch.inference_mode()
def run_stage1_denoise(
    transformer: torch.nn.Module,
    scheduler: object,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    latent_shape: Tuple[int, int, int, int, int],
    indices_hidden_states: torch.Tensor,
    indices_latents_history_short: torch.Tensor,
    indices_latents_history_mid: torch.Tensor,
    indices_latents_history_long: torch.Tensor,
    latents_history_short: torch.Tensor,
    latents_history_mid: torch.Tensor,
    latents_history_long: torch.Tensor,
    num_inference_steps: int,
    seed: int,
) -> torch.Tensor:
    """执行单个 section 的扩散式去噪恢复。"""
    base_transformer = unwrap_model(transformer)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, mu=1.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    # 从纯噪声初始化，逐 timestep 向恢复结果逼近。
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32, generator=generator)

    for timestep in scheduler.timesteps:
        timestep_batch = timestep.expand(latents.shape[0])
        latent_model_input = latents.to(weight_dtype)
        with base_transformer.cache_context("cond"):
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short.to(weight_dtype),
                latents_history_mid=latents_history_mid.to(weight_dtype),
                latents_history_long=latents_history_long.to(weight_dtype),
                return_dict=False,
            )[0]
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    if hasattr(base_transformer, "clear_kv_cache"):
        # 长序列逐段推理后清 cache，避免显存逐步累积。
        base_transformer.clear_kv_cache()
    return latents.cpu()


def build_low_latent_payload(sequence: PreparedSequence) -> Dict[str, object]:
    """构造 low_latents 的落盘 payload。"""
    return {
        "format_version": LOW_LATENT_FORMAT_V1,
        "input_path": str(sequence.path),
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
        "chunk_lengths": list(sequence.metadata.chunk_lengths),
        "codec_config": sequence.low_codec_payload["codec_config"],
        "keyframe": sequence.low_codec_payload["keyframe"],
        "quantized_remainder": sequence.low_codec_payload["quantized_remainder"],
        "scales": sequence.low_codec_payload["scales"],
        "reduced_shape": list(sequence.low_codec_payload["reduced_shape"]),
        "original_remainder_shape": list(sequence.low_codec_payload["original_remainder_shape"]),
        "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
        "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
    }


def build_recover_latent_payload(
    sequence: PreparedSequence,
    recovered_full_latents: torch.Tensor,
    base_model_path: str,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    """构造 recover_latents 的落盘 payload。

    恢复后的 full latents 会被重新切回 chunk 形式，方便继续复用项目里既有的 latent IO 逻辑。
    """
    restored_chunks = split_full_latents(recovered_full_latents, sequence.metadata.chunk_lengths)
    return {
        "format_version": LATENT_FORMAT_V2,
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
        "chunk_frame_ranges": [list(frame_range) for frame_range in sequence.metadata.chunk_frame_ranges],
        "latent_chunks": restored_chunks,
        "model_path": str(base_model_path),
        "recovery_metadata": {
            "input_path": str(sequence.path),
            "checkpoint_dir": str(checkpoint_dir),
            "raw_bpp": sequence.metadata.raw_bpp,
            "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
            "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
        },
    }


def resolve_output_paths(
    input_path: Path,
    input_root: Path,
    low_dir: Path,
    recover_dir: Path,
    metrics_dir: Path,
) -> Tuple[Path, Path, Path]:
    """为 low_latents、recover_latents 和 metrics 生成对应输出路径。"""
    relative_path = input_path.relative_to(input_root)
    low_latent_path = (low_dir / relative_path).with_suffix(".pt")
    recover_latent_path = (recover_dir / relative_path).with_suffix(".pt")
    metrics_path = (metrics_dir / relative_path).with_suffix(".json")
    low_latent_path.parent.mkdir(parents=True, exist_ok=True)
    recover_latent_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    return low_latent_path, recover_latent_path, metrics_path


def compute_tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """计算 latent 级别的基础误差指标。"""
    diff = (prediction.float() - target.float()).flatten()
    mse = float(diff.pow(2).mean().item())
    l1 = float(diff.abs().mean().item())
    psnr = float("inf") if mse <= EPS else float(-10.0 * math.log10(mse + EPS))
    return {
        "mse": mse,
        "l1": l1,
        "psnr": psnr,
    }


def compute_boundary_transition_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent_window_size: int,
) -> float:
    """计算 section 边界处的时序跳变误差。

    由于恢复是分窗口进行的，边界位置最容易出现突变。
    这里比较边界前后两帧的差分误差，作为简单的时序连续性指标。
    """
    boundary_errors = []
    for section_start in range(1 + latent_window_size, prediction.shape[1], latent_window_size):
        pred_delta = prediction[:, section_start] - prediction[:, section_start - 1]
        target_delta = target[:, section_start] - target[:, section_start - 1]
        boundary_errors.append((pred_delta - target_delta).abs().mean().item())
    if not boundary_errors:
        return 0.0
    return float(sum(boundary_errors) / len(boundary_errors))


def save_training_artifacts(
    output_dir: Path,
    transformer: torch.nn.Module,
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    args: argparse.Namespace,
    train_metrics: Dict[str, object],
    accelerator=None,
) -> None:
    """保存 recover 训练产物。

    这里统一持久化三类信息：
    - `transformer_full.pt`: 恢复模型权重
    - `recover_config.json`: 推理必须依赖的结构化配置
    - `train_metrics.json`: 训练过程指标，便于对比不同实验
    """
    if accelerator is None or accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    save_transformer_checkpoint(transformer, output_dir / "transformer_full.pt", accelerator=accelerator)

    # recover_config 只保留推理真正需要的最小配置，避免把一次性训练细节全部耦合进去。
    recover_config = {
        "format_version": RECOVER_CONFIG_VERSION,
        "base_model_path": args.base_model_path,
        "weight_dtype": args.weight_dtype,
        "history_sizes": list(history_sizes),
        "latent_window_size": latent_window_size,
        "codec_config": asdict(codec_config),
        "train_loss_config": {
            "loss_weighting_scheme": args.loss_weighting_scheme,
            "logit_mean": args.logit_mean,
            "logit_std": args.logit_std,
            "mode_scale": args.mode_scale,
            "flow_loss_weight": args.flow_loss_weight,
            "x_loss_weight": args.x_loss_weight,
            "noise_loss_weight": args.noise_loss_weight,
            "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
            "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        },
    }
    if accelerator is None or accelerator.is_main_process:
        with (output_dir / "recover_config.json").open("w", encoding="utf-8") as handle:
            json.dump(recover_config, handle, indent=2)
        with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(train_metrics, handle, indent=2)

    if accelerator is not None:
        accelerator.wait_for_everyone()


def save_transformer_checkpoint(transformer: torch.nn.Module, output_path: Path, accelerator=None) -> None:
    """把 transformer 权重转到 CPU 后保存，减少 checkpoint 与设备环境的耦合。"""
    if accelerator is not None:
        state_dict = accelerator.get_state_dict(transformer)
        if not accelerator.is_main_process:
            return
    else:
        state_dict = transformer.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.detach().cpu() if torch.is_tensor(value) else value
    torch.save(cpu_state_dict, output_path)


def load_recover_config(checkpoint_dir: Path) -> Dict[str, object]:
    """读取并校验 recover checkpoint 的配置文件。"""
    config_path = checkpoint_dir / "recover_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing recover config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_keys = {"format_version", "history_sizes", "latent_window_size", "codec_config"}
    missing_keys = sorted(required_keys - config.keys())
    if missing_keys:
        raise KeyError(f"Recover config {config_path} is missing keys: {missing_keys}")
    if config["format_version"] != RECOVER_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported recover config format in {config_path}: {config['format_version']}. "
            f"Expected {RECOVER_CONFIG_VERSION}."
        )
    return config


def resolve_infer_value(cli_value: object, default_value: object, checkpoint_value: object) -> object:
    """解析推理参数的优先级。

    优先级规则为：
    - 若 checkpoint 没存该值，则使用 CLI 值
    - 若用户显式传了非默认 CLI 值，则尊重 CLI
    - 否则沿用 checkpoint 中训练时保存的值
    """
    if checkpoint_value is None:
        return cli_value
    if cli_value != default_value:
        return cli_value
    return checkpoint_value


def load_state_dict_file(path: Path) -> Dict[str, torch.Tensor]:
    """兼容不同 PyTorch 版本加载 state dict。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_weight_dtype(weight_dtype: str) -> torch.dtype:
    """把字符串形式的 dtype 解析为 torch.dtype。"""
    if weight_dtype == "bf16":
        return torch.bfloat16
    if weight_dtype == "fp16":
        return torch.float16
    if weight_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported weight dtype: {weight_dtype}")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """统计模型中需要训练的参数量。"""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def init_distributed_context(device_arg: str) -> Tuple[DistributedContext, torch.device]:
    """初始化推理阶段的分布式上下文。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(is_distributed=False), resolve_device(device_arg)

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    rank = int(os.environ.get("RANK", "0"))
    if local_rank < 0:
        raise RuntimeError("Distributed mode requires LOCAL_RANK to be set. Please launch with torchrun.")

    requested_device = torch.device(device_arg)
    if requested_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA distributed training was requested but CUDA is not available.")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = resolve_device(device_arg)
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return (
        DistributedContext(
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
        ),
        device,
    )


def cleanup_distributed_context(distributed_context: DistributedContext) -> None:
    """安全释放分布式进程组。"""
    if distributed_context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed_context: DistributedContext) -> None:
    """在分布式推理中做同步屏障。"""
    if distributed_context.is_distributed and dist.is_initialized():
        dist.barrier()


def reduce_tensor_mean(tensor: torch.Tensor, distributed_context: DistributedContext) -> torch.Tensor:
    """对分布式 tensor 做均值归约。

    当前文件里几乎没有直接使用它，但保留这个工具函数便于后续扩展更多分布式指标。
    """
    if not distributed_context.is_distributed:
        return tensor.detach().float()
    reduced = tensor.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= distributed_context.world_size
    return reduced


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """剥离 DDP / accelerate 外层包装，拿到真正的底层模型。"""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        next_model = getattr(unwrapped, "module")
        if next_model is unwrapped:
            break
        unwrapped = next_model
    return unwrapped


def set_seed(seed: int) -> None:
    """统一设置 Python / PyTorch / CUDA 随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
