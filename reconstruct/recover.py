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
import inspect
import json
import math
import os
import random
import shutil
import sys
import time
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from reconstruct.codec_gop import (
    TRILINEAR_TAIL_CODEC_TYPE,
    decode_low_latents_payload,
    encode_anchor_plus_tail_latents,
    estimate_anchor_plus_tail_codec_bytes,
    make_section_ranges,
    validate_codec_config,
)
from reconstruct.learned_codec import LEARNED_TAIL_CODEC_TYPE, build_learned_tail_codec
from reconstruct.latent_io import (
    DEFAULT_BASE_MODEL_PATH,
    LATENT_FORMAT_V2,
    LOW_LATENT_FORMAT_V2,
    LOW_LATENT_FORMAT_V3,
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
DEFAULT_TEMPORAL_FACTOR = 2
DEFAULT_SPATIAL_FACTOR = 4
DEFAULT_QUANT_DTYPE = "int8"
DEFAULT_KEYFRAME_DTYPE = "float16"
DEFAULT_HISTORY_SIZES = [3, 1, 1]
DEFAULT_LATENT_WINDOW_SIZE = 3  # Helios的VAE中4帧视频压缩成一个latent时间步，所以其实一个latent对应4帧,最终对应原视频的关系是 (DEFAULT_ANCHOR_SPAN_LATENTS + (DEFAULT_LATENT_WINDOW_SIZE - 1) * 4 = 原视频帧数)
DEFAULT_SECTION_SPAN_LATENTS = DEFAULT_LATENT_WINDOW_SIZE
DEFAULT_ANCHOR_SPAN_LATENTS = 1
DEFAULT_TAIL_SPAN_LATENTS = DEFAULT_SECTION_SPAN_LATENTS - DEFAULT_ANCHOR_SPAN_LATENTS
DEFAULT_ANCHOR_QUANT_DTYPE = "int8"
DEFAULT_ANCHOR_SPATIAL_FACTOR = 1
DEFAULT_TAIL_CODEC_TYPE = LEARNED_TAIL_CODEC_TYPE
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
DEFAULT_TEMPORAL_DELTA_LOSS_WEIGHT = 0.1
DEFAULT_FIX_ANCHOR_DURING_DENOISE = True
DEFAULT_LOW_LATENT_FORMAT_VERSION = LOW_LATENT_FORMAT_V3
RECOVER_CONFIG_VERSION = "helios_recover_v5"
CURRENT_LOW_TARGET_MODE_FULL_WINDOW_1X = "full_window_1x"
CURRENT_LOW_TARGET_INIT_PATCH_SHORT = "patch_short"
LEARNED_CODEC_CHECKPOINT_NAME = "learned_codec.pt"
LATEST_CHECKPOINT_LINK_NAME = "latest"
LATEST_CHECKPOINT_POINTER_NAME = "latest_checkpoint.txt"
EPS = 1e-8


@dataclass
class CodecConfig:
    """低码率 latent 编码配置。

    Recover V2 的 codec 不再把首帧后的 remainder 视为单一流，而是拆成：
    - `global_keyframe(x0)`：全局唯一高精度锚点
    - `local refresh anchor`：每个 section 的局部刷新锚点
    - `P-tail residual`：相对 local anchor 的尾部残差表示
    """

    temporal_factor: int = DEFAULT_TEMPORAL_FACTOR
    spatial_factor: int = DEFAULT_SPATIAL_FACTOR
    quant_dtype: str = DEFAULT_QUANT_DTYPE
    keyframe_dtype: str = DEFAULT_KEYFRAME_DTYPE
    section_span_latents: int = DEFAULT_SECTION_SPAN_LATENTS
    anchor_span_latents: int = DEFAULT_ANCHOR_SPAN_LATENTS
    tail_span_latents: int = DEFAULT_TAIL_SPAN_LATENTS
    anchor_quant_dtype: str = DEFAULT_ANCHOR_QUANT_DTYPE
    anchor_spatial_factor: int = DEFAULT_ANCHOR_SPATIAL_FACTOR
    tail_codec_type: str = DEFAULT_TAIL_CODEC_TYPE
    learned_codec_hidden_channels: Optional[int] = None


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
    low_codec_payload: Optional[Dict[str, object]] = None
    low_full_latents: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DistributedContext:
    """推理阶段的分布式上下文。

    这里主要服务 `infer` 命令：把不同输入 latent 文件分摊到多个 rank，
    每个 rank 独立完成各自样本的低码率解码、恢复与落盘。
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
    temporal_delta_loss: torch.Tensor
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
            "temporal_delta_loss": self.temporal_delta_loss.detach().float(),
            "aux_scale": self.aux_scale.detach().float(),
            "sigma_mean": self.sigma_mean.detach().float(),
            "flow_target_rms": self.flow_target_rms.detach().float(),
            "flow_pred_rms": self.flow_pred_rms.detach().float(),
            "x_target_rms": self.x_target_rms.detach().float(),
            "x_pred_rms": self.x_pred_rms.detach().float(),
            "noise_target_rms": self.noise_target_rms.detach().float(),
            "noise_pred_rms": self.noise_pred_rms.detach().float(),
        }


class RecoverTrainingBundle(torch.nn.Module):
    """把 recover transformer 与 learned tail codec 组合成单一训练模块。

    这样训练阶段可以继续沿用单模型的 Accelerate / DeepSpeed 主流程，
    避免把压缩网络与恢复网络拆成两个独立 prepare 对象。
    """

    def __init__(self, transformer: torch.nn.Module, learned_tail_codec: Optional[torch.nn.Module]):
        super().__init__()
        self.transformer = transformer
        self.learned_tail_codec = learned_tail_codec


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
        anchor_span_latents: int,
    ):
        self.sequences = list(sequences)
        self.history_sizes = list(history_sizes)
        self.history_window_size = sum(history_sizes)
        self.latent_window_size = latent_window_size
        self.anchor_span_latents = anchor_span_latents
        self.samples: List[Tuple[int, int]] = []

        for seq_idx, sequence in enumerate(self.sequences):
            total_latent_frames = sequence.clean_full_latents.shape[1]
            # section_start 从 1 开始，意味着首帧默认单独保留为 keyframe，
            # 后续所有帧都交给 recover 模块按窗口学习恢复。
            for start in range(1, total_latent_frames, latent_window_size):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, int]:
        seq_idx, section_start = self.samples[index]
        return {
            "seq_idx": seq_idx,
            "section_start": section_start,
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
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Training output directory. Please pass from launch script.",
    )
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
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
    parser.add_argument("--section_span_latents", type=int, default=None)
    parser.add_argument("--anchor_span_latents", type=int, default=DEFAULT_ANCHOR_SPAN_LATENTS)
    parser.add_argument("--tail_span_latents", type=int, default=None)
    parser.add_argument("--anchor_quant_dtype", type=str, default=DEFAULT_ANCHOR_QUANT_DTYPE, choices=["int8"])
    parser.add_argument("--anchor_spatial_factor", type=int, default=DEFAULT_ANCHOR_SPATIAL_FACTOR)
    parser.add_argument(
        "--tail_codec_type",
        type=str,
        default=DEFAULT_TAIL_CODEC_TYPE,
        choices=[TRILINEAR_TAIL_CODEC_TYPE, LEARNED_TAIL_CODEC_TYPE],
    )
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
    parser.add_argument("--temporal_delta_loss_weight", type=float, default=DEFAULT_TEMPORAL_DELTA_LOSS_WEIGHT)
    parser.add_argument("--aux_loss_warmup_steps", type=int, default=DEFAULT_AUX_LOSS_WARMUP_STEPS)
    parser.add_argument("--aux_loss_ramp_steps", type=int, default=DEFAULT_AUX_LOSS_RAMP_STEPS)
    parser.add_argument(
        "--fix_anchor_during_denoise",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIX_ANCHOR_DURING_DENOISE,
    )


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    """注册推理阶段使用的参数。"""
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Inference output directory. Please pass from launch script.",
    )
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
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only run inference on the first N latent files before distributed sharding.",
    )


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


def import_schedule_shift_fn():
    """导入 Helios 的动态调度偏移函数。"""
    from helios.utils.utils_base import apply_schedule_shift

    return apply_schedule_shift


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
    if args.temporal_delta_loss_weight < 0.0:
        raise ValueError(
            f"temporal_delta_loss_weight must be >= 0, got {args.temporal_delta_loss_weight}."
        )
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
            "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
            "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
            "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
            "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        },
        reinit=True,
    )


def build_periodic_checkpoint_dir(output_dir: Path, epoch: int, global_step: int) -> Path:
    """为阶段性 checkpoint 生成稳定目录名，便于恢复与对比不同训练阶段效果。"""
    return output_dir / "checkpoints" / f"epoch_{epoch:04d}_step_{global_step:08d}"


def build_temporary_checkpoint_dir(output_dir: Path) -> Path:
    """为原子化 checkpoint 发布生成隐藏的临时目录。"""
    stamp = f"{os.getpid()}_{time.time_ns()}"
    return output_dir.parent / f".{output_dir.name}.tmp_{stamp}"


def build_atomic_temp_path(output_path: Path) -> Path:
    """为单文件原子写入生成同目录下的临时路径。"""
    stamp = f"{os.getpid()}_{time.time_ns()}"
    return output_path.parent / f".{output_path.name}.tmp_{stamp}"


def is_managed_checkpoint_dir(output_dir: Path) -> bool:
    """判断目标目录是否属于 `checkpoints/` 下的阶段性 checkpoint。"""
    return output_dir.parent.name == "checkpoints"


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
    latent_window_size = codec_config.section_span_latents
    history_sizes = normalize_history_sizes(args.history_sizes)
    input_root, latent_paths = discover_input_latent_paths(args.input_path)
    # 联合优化时训练阶段不预先固化 low latents，而是保留 clean latents，后续在线压缩并反传。
    prepared_sequences = [
        prepare_sequence(
            path,
            codec_config,
            learned_tail_codec=None,
            materialize_low_latents=False,
        )
        for path in latent_paths
    ]
    dataset = LatentWindowDataset(
        sequences=prepared_sequences,
        history_sizes=history_sizes,
        latent_window_size=latent_window_size,
        anchor_span_latents=codec_config.anchor_span_latents,
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
    learned_tail_codec = None
    learned_codec_trainable_params = 0
    if codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE:
        learned_tail_codec = build_learned_tail_codec(
            codec_config=asdict(codec_config),
            in_channels=infer_latent_channels(prepared_sequences),
        ).to(device)
        learned_tail_codec.train()
        learned_codec_trainable_params = count_trainable_parameters(learned_tail_codec)
    # transformer 继续走 Accelerate/DeepSpeed；learned codec 保持各 rank 本地副本，手动同步梯度。
    trainable_params = count_trainable_parameters(transformer) + learned_codec_trainable_params

    optimizer = torch.optim.AdamW(
        [param for param in transformer.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    learned_codec_optimizer = None
    if learned_tail_codec is not None:
        learned_codec_optimizer = torch.optim.AdamW(
            [param for param in learned_tail_codec.parameters() if param.requires_grad],
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
    synchronize_module_parameters(learned_tail_codec)

    output_dir = args.output_dir.resolve()
    checkpoint_save_interval_epochs = max(1, math.ceil(args.epochs / 5)) # 这里记录了多少个epoch保存一下权重
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[train] input_root={input_root} files={len(latent_paths)} windows={len(dataset)} "
            f"device={device} world_size={accelerator.num_processes} "
            f"effective_global_batch_size={args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
            f"tail_codec_type={codec_config.tail_codec_type} "
            f"save_every_epochs={checkpoint_save_interval_epochs} "
            f"loss_weighting_scheme={args.loss_weighting_scheme} "
            f"flow/x/noise/delta=({args.flow_loss_weight:.3f}/{args.x_loss_weight:.3f}/{args.noise_loss_weight:.3f}/{args.temporal_delta_loss_weight:.3f}) "
            f"aux_warmup={args.aux_loss_warmup_steps} aux_ramp={args.aux_loss_ramp_steps}"
        )
    accelerator.wait_for_everyone()

    train_metrics: Dict[str, object] = {
        "checkpoint_format_version": RECOVER_CONFIG_VERSION,
        "losses": [],
        "steps": 0,
        "trainable_params": trainable_params,
        "learned_codec_trainable_params": learned_codec_trainable_params,
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
        "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
        "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
        "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        "tail_codec_type": codec_config.tail_codec_type,
        "loss_definition": {
            "optimized_loss": (
                "flow_loss_weight * tail_position_weighted_mse(flow_pred, noise-latent) + "
                "aux_scale * (x_loss_weight * tail_position_weighted_normalized_mse(x_pred, latent) + "
                "noise_loss_weight * tail_position_weighted_normalized_mse(noise_pred, noise) + "
                "temporal_delta_loss_weight * tail_position_weighted_l1(delta(x_pred), delta(latent)))"
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
        "temporal_delta_losses": [],
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
                latent_window_size=latent_window_size,
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
                "temporal_delta_loss": 0.0,
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
                        sequences=prepared_sequences,
                        codec_config=codec_config,
                        transformer=transformer,
                        learned_tail_codec=learned_tail_codec,
                        prompt_embeds=prompt_embeds,
                        device=device,
                        weight_dtype=weight_dtype,
                        history_sizes=history_sizes,
                        latent_window_size=latent_window_size,
                        anchor_span_latents=codec_config.anchor_span_latents,
                        loss_weighting_scheme=args.loss_weighting_scheme,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                        flow_loss_weight=args.flow_loss_weight,
                        x_loss_weight=args.x_loss_weight,
                        noise_loss_weight=args.noise_loss_weight,
                        temporal_delta_loss_weight=args.temporal_delta_loss_weight,
                        aux_loss_warmup_steps=args.aux_loss_warmup_steps,
                        aux_loss_ramp_steps=args.aux_loss_ramp_steps,
                        global_step=global_step,
                    )
                    accelerator.backward(step_output.loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                        if learned_tail_codec is not None:
                            synchronize_module_gradients(learned_tail_codec)
                            torch.nn.utils.clip_grad_norm_(learned_tail_codec.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if learned_codec_optimizer is not None and accelerator.sync_gradients:
                        learned_codec_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    if learned_codec_optimizer is not None and accelerator.sync_gradients:
                        learned_codec_optimizer.zero_grad(set_to_none=True)

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
                    temporal_delta_loss_value = scalar_metrics["temporal_delta_loss"]
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
                    train_metrics["temporal_delta_losses"].append(temporal_delta_loss_value)
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
                    epoch_metric_sums["temporal_delta_loss"] += temporal_delta_loss_value
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
                            delta=f"{temporal_delta_loss_value:.6f}",
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
                                "train/temporal_delta_loss": temporal_delta_loss_value,
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
                    "temporal_delta_loss_mean": epoch_metric_sums["temporal_delta_loss"] / epoch_logged_steps,
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
                            "epoch/temporal_delta_loss_mean": epoch_summary["temporal_delta_loss_mean"],
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
                    learned_tail_codec=learned_tail_codec,
                    training_model=None,
                    codec_config=codec_config,
                    history_sizes=history_sizes,
                    latent_window_size=latent_window_size,
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
            learned_tail_codec=learned_tail_codec,
            training_model=None,
            codec_config=codec_config,
            history_sizes=history_sizes,
            latent_window_size=latent_window_size,
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
                wandb_run.summary["train/last_temporal_delta_loss"] = train_metrics["temporal_delta_losses"][-1]
                wandb_run.summary["train/last_loss_ema"] = train_metrics["losses_ema"][-1]
                wandb_run.summary["train/trainable_params"] = train_metrics["trainable_params"]
            print(
                f"[train] completed steps={train_metrics['steps']} "
                f"last_loss={train_metrics['losses'][-1]:.6f} "
                f"last_loss_ema={train_metrics['losses_ema'][-1]:.6f} "
                f"last_flow_loss={train_metrics['flow_losses'][-1]:.6f} "
                f"last_x_loss={train_metrics['x_losses'][-1]:.6f} "
                f"last_noise_loss={train_metrics['noise_losses'][-1]:.6f} "
                f"last_temporal_delta_loss={train_metrics['temporal_delta_losses'][-1]:.6f} "
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
        requested_checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint_dir = resolve_checkpoint_dir_for_inference(requested_checkpoint_dir)
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
        latent_window_size = int(checkpoint_config.get("section_span_latents", checkpoint_config["latent_window_size"]))
        anchor_span_latents = int(checkpoint_config.get("anchor_span_latents", codec_config.anchor_span_latents))
        fix_anchor_during_denoise = bool(
            checkpoint_config.get("fix_anchor_during_denoise", DEFAULT_FIX_ANCHOR_DURING_DENOISE)
        )
        weight_dtype = parse_weight_dtype(str(weight_dtype_name))

        input_root, latent_paths = discover_input_latent_paths(args.input_path)
        total_input_files = len(latent_paths)
        if args.max_samples is not None:
            if args.max_samples <= 0:
                raise ValueError(f"--max_samples must be a positive integer, got {args.max_samples}.")
            latent_paths = latent_paths[: args.max_samples]
        prototype_sequence = prepare_sequence(
            latent_paths[0],
            codec_config,
            learned_tail_codec=None,
            materialize_low_latents=False,
        )
        learned_tail_codec = load_learned_tail_codec(
            checkpoint_dir=checkpoint_dir,
            codec_config=codec_config,
            latent_channels=int(prototype_sequence.clean_full_latents.shape[0]),
            device=device,
        )
        assigned_latent_paths = latent_paths[distributed_context.rank :: distributed_context.world_size]
        output_dir = args.output_dir.resolve()
        low_dir = output_dir / "low_latents"
        recover_dir = output_dir / "recover_latents"
        metrics_dir = output_dir / "metrics"

        if distributed_context.is_main_process:
            low_dir.mkdir(parents=True, exist_ok=True)
            recover_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[infer] input_root={input_root} files={len(latent_paths)}/{total_input_files} "
                f"device={device} world_size={distributed_context.world_size} "
                f"checkpoint_dir={checkpoint_dir}"
            )
            if checkpoint_dir != requested_checkpoint_dir:
                print(f"[infer] resolved checkpoint request {requested_checkpoint_dir} -> {checkpoint_dir}")
        distributed_barrier(distributed_context)

        transformer, scheduler, prompt_embeds = load_transformer_bundle(
            base_model_path=str(base_model_path),
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_dir=checkpoint_dir,
            gradient_checkpointing=False,
        )
        transformer.eval()

        for sample_idx, latent_path in enumerate(assigned_latent_paths, start=1):
            print(
                f"[infer][rank={distributed_context.rank}] ({sample_idx}/{len(assigned_latent_paths)}) "
                f"input={latent_path}"
            )

            sequence = prepare_sequence(
                latent_path,
                codec_config,
                learned_tail_codec=learned_tail_codec,
                materialize_low_latents=True,
                use_ste_quant=False,
                move_to_cpu=True,
            )
            recovered_full_latents = reconstruct_sequence(
                sequence=sequence,
                transformer=transformer,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                device=device,
                weight_dtype=weight_dtype,
                history_sizes=history_sizes,
                latent_window_size=latent_window_size,
                anchor_span_latents=anchor_span_latents,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                fix_anchor_during_denoise=fix_anchor_during_denoise,
                distributed_context=distributed_context,
            )
            low_latent_path, recover_latent_path, metrics_path = resolve_output_paths(
                input_path=latent_path,
                input_root=input_root,
                low_dir=low_dir,
                recover_dir=recover_dir,
                metrics_dir=metrics_dir,
            )
            low_payload = build_low_latent_payload(sequence)
            torch.save(low_payload, low_latent_path)
            ensure_finite_recover_state(
                stage="final_output",
                input_path=sequence.path,
                section_start=None,
                step_idx=None,
                timestep=None,
                sigma=None,
                latents=recovered_full_latents,
            )
            recover_payload = build_recover_latent_payload(
                sequence=sequence,
                recovered_full_latents=recovered_full_latents,
                base_model_path=str(base_model_path),
                checkpoint_dir=checkpoint_dir,
            )
            torch.save(recover_payload, recover_latent_path)

            direct_low_metrics = compute_tensor_metrics(sequence.low_full_latents, sequence.clean_full_latents)
            restored_metrics = compute_tensor_metrics(recovered_full_latents, sequence.clean_full_latents)
            low_tail_position_metrics = compute_tail_position_metrics(
                prediction=sequence.low_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
                anchor_span_latents=anchor_span_latents,
            )
            recover_tail_position_metrics = compute_tail_position_metrics(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
                anchor_span_latents=anchor_span_latents,
            )
            tail_only_metrics = compute_anchor_tail_metrics(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
                anchor_span_latents=anchor_span_latents,
                mode="tail",
            )
            anchor_reconstruction_metrics = compute_anchor_tail_metrics(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
                anchor_span_latents=anchor_span_latents,
                mode="anchor",
            )
            temporal_delta_l1 = compute_temporal_delta_l1(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
            )
            low_boundary_transition_l1 = compute_boundary_transition_l1(
                prediction=sequence.low_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
            )
            recover_boundary_transition_l1 = compute_boundary_transition_l1(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
                latent_window_size=latent_window_size,
            )
            temporal_backtrack_metrics = compute_temporal_backtrack_metrics(
                prediction=recovered_full_latents,
                target=sequence.clean_full_latents,
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
                "low_tail_position_metrics": low_tail_position_metrics,
                "recover_tail_position_metrics": recover_tail_position_metrics,
                "tail_only_metrics": tail_only_metrics,
                "anchor_reconstruction_metrics": anchor_reconstruction_metrics,
                "temporal_delta_l1": temporal_delta_l1,
                "low_boundary_transition_l1": low_boundary_transition_l1,
                "recover_boundary_transition_l1": recover_boundary_transition_l1,
                "section_boundary_delta_l1": recover_boundary_transition_l1,
                "boundary_transition_l1": recover_boundary_transition_l1,
                **temporal_backtrack_metrics,
                "codec_config": asdict(codec_config),
                "checkpoint_dir": str(checkpoint_dir),
                "num_inference_steps": args.num_inference_steps,
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)

            print(
                f"[infer][rank={distributed_context.rank}] saved low={low_latent_path} recover={recover_latent_path} "
                f"low_bpp={sequence.low_codec_payload['low_bpp']:.6f} "
                f"direct_low_l1={direct_low_metrics['l1']:.6f} restored_l1={restored_metrics['l1']:.6f}"
            )

            if device.type == "cuda":
                # 逐样本清 cache，减轻长序列推理时的显存峰值压力。
                torch.cuda.empty_cache()
        distributed_barrier(distributed_context)
    finally:
        cleanup_distributed_context(distributed_context)


def build_codec_config(args: argparse.Namespace) -> CodecConfig:
    """从命令行参数构造低码率编码配置。"""
    section_span_latents = (
        int(args.section_span_latents) if getattr(args, "section_span_latents", None) is not None else int(args.latent_window_size)
    )
    anchor_span_latents = int(getattr(args, "anchor_span_latents", DEFAULT_ANCHOR_SPAN_LATENTS))
    if getattr(args, "tail_span_latents", None) is None:
        tail_span_latents = section_span_latents - anchor_span_latents
    else:
        tail_span_latents = int(args.tail_span_latents)

    codec_config = CodecConfig(
        temporal_factor=args.temporal_factor,
        spatial_factor=args.spatial_factor,
        quant_dtype=args.quant_dtype,
        keyframe_dtype=args.keyframe_dtype,
        section_span_latents=section_span_latents,
        anchor_span_latents=anchor_span_latents,
        tail_span_latents=tail_span_latents,
        anchor_quant_dtype=args.anchor_quant_dtype,
        anchor_spatial_factor=args.anchor_spatial_factor,
        tail_codec_type=args.tail_codec_type,
    )
    validate_codec_config(asdict(codec_config))
    return codec_config


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


def prepare_sequence(
    path: Path,
    codec_config: CodecConfig,
    learned_tail_codec: Optional[torch.nn.Module] = None,
    materialize_low_latents: bool = True,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
) -> PreparedSequence:
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
    sequence = PreparedSequence(
        path=path.resolve(),
        metadata=metadata,
        clean_full_latents=clean_full_latents,
    )
    if not materialize_low_latents:
        return sequence

    low_codec_payload, low_full_latents = build_sequence_low_latents(
        sequence=sequence,
        codec_config=codec_config,
        learned_tail_codec=learned_tail_codec,
        device=None,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
    )
    sequence.low_codec_payload = low_codec_payload
    sequence.low_full_latents = low_full_latents
    return sequence


def infer_latent_channels(sequences: Sequence[PreparedSequence]) -> int:
    if not sequences:
        raise ValueError("Expected at least one sequence to infer latent channels.")
    latent_channels = int(sequences[0].clean_full_latents.shape[0])
    for sequence in sequences[1:]:
        current_channels = int(sequence.clean_full_latents.shape[0])
        if current_channels != latent_channels:
            raise ValueError(
                "All sequences must share the same latent channel count for a shared learned codec, "
                f"got {latent_channels} and {current_channels}."
            )
    return latent_channels


def build_sequence_low_latents(
    sequence: PreparedSequence,
    codec_config: CodecConfig,
    learned_tail_codec: Optional[torch.nn.Module],
    device: Optional[torch.device],
    use_ste_quant: bool,
    move_to_cpu: bool,
) -> Tuple[Dict[str, object], torch.Tensor]:
    clean_full_latents = sequence.clean_full_latents
    if device is None and learned_tail_codec is not None:
        try:
            device = next(learned_tail_codec.parameters()).device
        except StopIteration:
            device = None
    if device is not None:
        clean_full_latents = clean_full_latents.to(device=device, dtype=torch.float32)
    else:
        clean_full_latents = clean_full_latents.float()

    # 压缩到低码率模式，模拟真实传输时候的码率
    low_codec_payload = encode_low_latents(
        clean_full_latents=clean_full_latents,
        codec_config=codec_config,
        chunk_lengths=sequence.metadata.chunk_lengths,
        learned_tail_codec=learned_tail_codec,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
    )
    low_codec_payload["low_codec_bytes"] = estimate_low_codec_bytes(low_codec_payload)
    low_codec_payload["low_bpp"] = compute_bpp_from_total_pixels(
        int(low_codec_payload["low_codec_bytes"]),
        sequence.metadata.total_pixels,
    )
    # 讲latents恢复到原尺寸
    low_full_latents = decode_low_latents(
        codec_payload=low_codec_payload,
        learned_tail_codec=learned_tail_codec,
    ).float().contiguous()
    if move_to_cpu:
        low_full_latents = low_full_latents.cpu().contiguous()
    return low_codec_payload, low_full_latents


def encode_low_latents(
    clean_full_latents: torch.Tensor,
    codec_config: CodecConfig,
    chunk_lengths: Sequence[int],
    learned_tail_codec: Optional[torch.nn.Module] = None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    """把 clean latents 编码成 `x0 + local anchor + P-tail residual` 低码率载荷。"""
    codec_payload = encode_anchor_plus_tail_latents(
        clean_full_latents=clean_full_latents,
        codec_config=asdict(codec_config),
        chunk_lengths=chunk_lengths,
        learned_tail_codec=learned_tail_codec,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
    )
    codec_payload["codec_config"] = asdict(codec_config)
    codec_payload["format_version"] = (
        DEFAULT_LOW_LATENT_FORMAT_VERSION
        if codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE
        else LOW_LATENT_FORMAT_V2
    )
    return codec_payload


def decode_low_latents(
    codec_payload: Dict[str, object],
    learned_tail_codec: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """从低码率载荷还原出粗恢复 latent。

    这个结果不是最终的 recover_latents，而是：
    - 训练时的历史条件输入
    - 推理时接收端可直接得到的低质量基线
    """
    return decode_low_latents_payload(codec_payload, learned_tail_codec=learned_tail_codec)


def estimate_low_codec_bytes(codec_payload: Dict[str, object]) -> int:
    """估算低码率载荷的字节数。

    这里统计的是 tensor 数据本身加最小必要 shape 元数据，
    用来近似衡量这条低码率链路的传输成本。
    """
    if "section_anchor_payloads" in codec_payload and "section_tail_payloads" in codec_payload:
        return estimate_anchor_plus_tail_codec_bytes(codec_payload)

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


def _extract_padded_latent_window(
    full_latents: torch.Tensor,
    section_start: int,
    latent_window_size: int,
    window_name: str,
) -> Tuple[torch.Tensor, int]:
    """截取一个 latent 窗口，并在末尾不足时重复最后一帧补齐。"""
    window = full_latents[:, section_start : section_start + latent_window_size]
    valid_frames = window.shape[1]
    if valid_frames == 0:
        raise ValueError(f"{window_name} at section_start={section_start} produced an empty window.")
    if valid_frames < latent_window_size:
        padding = window[:, -1:].repeat(1, latent_window_size - valid_frames, 1, 1)
        window = torch.cat([window, padding], dim=1)
    return window.contiguous(), valid_frames


def extract_target_window(
    clean_full_latents: torch.Tensor,
    section_start: int,
    latent_window_size: int,
) -> Tuple[torch.Tensor, int]:
    """从 clean latents 中截取一个目标恢复窗口。

    如果最后一个窗口长度不足 `latent_window_size`，则用最后一帧重复补齐；
    同时返回 `valid_target_frames`，后续通过 mask 避免 padding 帧影响 loss。
    """
    return _extract_padded_latent_window(
        full_latents=clean_full_latents,
        section_start=section_start,
        latent_window_size=latent_window_size,
        window_name="clean target window",
    )


def extract_current_low_target_window(
    low_full_latents: torch.Tensor,
    section_start: int,
    latent_window_size: int,
) -> Tuple[torch.Tensor, int]:
    """从 low latents 中截取当前 section 的完整 low window（anchor+tail）。"""
    return _extract_padded_latent_window(
        full_latents=low_full_latents,
        section_start=section_start,
        latent_window_size=latent_window_size,
        window_name="current low target window",
    )


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
            device=history.device,
            dtype=history.dtype,
        )
        history = torch.cat([zeros, history], dim=1)
    return history.contiguous()


def extract_section_anchor_window(
    low_full_latents: torch.Tensor,
    section_start: int,
    valid_target_frames: int,
    anchor_span_latents: int,
) -> torch.Tensor:
    anchor_steps = min(anchor_span_latents, valid_target_frames)
    anchor_latents = low_full_latents[:, section_start : section_start + anchor_steps]
    if anchor_latents.shape[1] == anchor_span_latents:
        return anchor_latents.contiguous()
    if anchor_latents.shape[1] == 0:
        return torch.zeros(
            low_full_latents.shape[0],
            anchor_span_latents,
            low_full_latents.shape[2],
            low_full_latents.shape[3],
            device=low_full_latents.device,
            dtype=low_full_latents.dtype,
        ).contiguous()
    padding = anchor_latents[:, -1:].repeat(1, anchor_span_latents - anchor_latents.shape[1], 1, 1)
    return torch.cat([anchor_latents, padding], dim=1).contiguous()


def build_training_batch_tensors(
    batch: Dict[str, torch.Tensor | int],
    sequences: Sequence[PreparedSequence],
    codec_config: CodecConfig,
    learned_tail_codec: Optional[torch.nn.Module],
    device: torch.device,
    history_sizes: Sequence[int],
    latent_window_size: int,
    anchor_span_latents: int,
) -> Dict[str, torch.Tensor]:
    history_window_size = sum(int(value) for value in history_sizes)
    seq_indices_value = batch["seq_idx"]
    section_starts_value = batch["section_start"]
    if torch.is_tensor(seq_indices_value):
        seq_indices = [int(value) for value in seq_indices_value.tolist()]
    else:
        seq_indices = [int(seq_indices_value)]
    if torch.is_tensor(section_starts_value):
        section_starts = [int(value) for value in section_starts_value.tolist()]
    else:
        section_starts = [int(section_starts_value)]

    unique_seq_indices = sorted(set(seq_indices))
    low_latent_cache: Dict[int, torch.Tensor] = {}
    clean_latent_cache: Dict[int, torch.Tensor] = {}
    for seq_idx in unique_seq_indices:
        sequence = sequences[seq_idx]
        _codec_payload, low_full_latents = build_sequence_low_latents(
            sequence=sequence,
            codec_config=codec_config,
            learned_tail_codec=learned_tail_codec,
            device=device,
            use_ste_quant=codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE,
            move_to_cpu=False,
        )
        clean_latent_cache[seq_idx] = sequence.clean_full_latents.to(device=device, dtype=torch.float32).contiguous()
        low_latent_cache[seq_idx] = low_full_latents.to(device=device, dtype=torch.float32).contiguous()

    history_latents: List[torch.Tensor] = []
    target_latents: List[torch.Tensor] = []
    current_low_target_latents: List[torch.Tensor] = []
    section_anchor_latents: List[torch.Tensor] = []
    x0_latents: List[torch.Tensor] = []
    valid_target_frames: List[int] = []
    for seq_idx, section_start in zip(seq_indices, section_starts):
        clean_full_latents = clean_latent_cache[seq_idx]
        low_full_latents = low_latent_cache[seq_idx]
        target_latent, valid_frames = extract_target_window(
            clean_full_latents=clean_full_latents,
            section_start=section_start,
            latent_window_size=latent_window_size,
        )
        current_low_target_latent, low_valid_frames = extract_current_low_target_window(
            low_full_latents=low_full_latents,
            section_start=section_start,
            latent_window_size=latent_window_size,
        )
        if low_valid_frames != valid_frames:
            raise RuntimeError(
                f"Current low target valid frames mismatch for seq_idx={seq_idx}, section_start={section_start}: "
                f"clean={valid_frames}, low={low_valid_frames}"
            )
        history_latent = extract_history_window(
            low_full_latents=low_full_latents,
            section_start=section_start,
            history_window_size=history_window_size,
        )
        anchor_latent = extract_section_anchor_window(
            low_full_latents=low_full_latents,
            section_start=section_start,
            valid_target_frames=valid_frames,
            anchor_span_latents=anchor_span_latents,
        )
        history_latents.append(history_latent)
        target_latents.append(target_latent)
        current_low_target_latents.append(current_low_target_latent)
        section_anchor_latents.append(anchor_latent)
        x0_latents.append(clean_full_latents[:, :1])
        valid_target_frames.append(valid_frames)

    return {
        "history_latents": torch.stack(history_latents, dim=0).contiguous(),
        "target_latents": torch.stack(target_latents, dim=0).contiguous(),
        "current_low_target_latents": torch.stack(current_low_target_latents, dim=0).contiguous(),
        "section_anchor_latents": torch.stack(section_anchor_latents, dim=0).contiguous(),
        "x0_latents": torch.stack(x0_latents, dim=0).contiguous(),
        "valid_target_frames": torch.tensor(valid_target_frames, device=device, dtype=torch.long),
    }


def initialize_current_low_target_branch_from_short(transformer: torch.nn.Module) -> None:
    patch_current = getattr(transformer, "patch_current_low_target", None)
    patch_short = getattr(transformer, "patch_short", None)
    if patch_current is None:
        raise RuntimeError(
            "Recover transformer is missing patch_current_low_target. "
            "Please ensure the current repo version is used for base model loading."
        )
    if patch_short is None:
        raise RuntimeError(
            "Recover transformer is missing patch_short, so patch_current_low_target cannot be initialized."
        )
    if patch_short.weight.is_meta:
        raise RuntimeError(
            "Recover transformer patch_short is still a meta tensor after base-model loading, "
            "so patch_current_low_target cannot be initialized from it."
        )
    if patch_current.weight.is_meta:
        patch_current.weight = torch.nn.Parameter(
            torch.empty_like(
                patch_short.weight,
                device=patch_short.weight.device,
                dtype=patch_short.weight.dtype,
            ),
            requires_grad=patch_short.weight.requires_grad,
        )
    if patch_current.bias is not None and patch_current.bias.is_meta:
        if patch_short.bias is None:
            raise RuntimeError(
                "Recover transformer patch_current_low_target has a bias parameter, but patch_short.bias is missing."
            )
        patch_current.bias = torch.nn.Parameter(
            torch.empty_like(
                patch_short.bias,
                device=patch_short.bias.device,
                dtype=patch_short.bias.dtype,
            ),
            requires_grad=patch_short.bias.requires_grad,
        )
    with torch.no_grad():
        patch_current.weight.copy_(patch_short.weight)
        if patch_current.bias is not None and patch_short.bias is not None:
            patch_current.bias.copy_(patch_short.bias)


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
            use_current_low_target_branch=True,
        )
        initialize_current_low_target_branch_from_short(transformer)
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
    sequences: Sequence[PreparedSequence],
    codec_config: CodecConfig,
    transformer: torch.nn.Module,
    learned_tail_codec: Optional[torch.nn.Module],
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    history_sizes: Sequence[int],
    latent_window_size: int,
    anchor_span_latents: int,
    loss_weighting_scheme: str,
    logit_mean: float,
    logit_std: float,
    mode_scale: float,
    flow_loss_weight: float,
    x_loss_weight: float,
    noise_loss_weight: float,
    temporal_delta_loss_weight: float,
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
    materialized_batch = build_training_batch_tensors(
        batch=batch,
        sequences=sequences,
        codec_config=codec_config,
        learned_tail_codec=learned_tail_codec,
        device=device,
        history_sizes=history_sizes,
        latent_window_size=latent_window_size,
        anchor_span_latents=anchor_span_latents,
    )
    history_latents = materialized_batch["history_latents"].to(dtype=weight_dtype)
    target_latents = materialized_batch["target_latents"].to(dtype=weight_dtype)
    current_low_target_latents = materialized_batch["current_low_target_latents"].to(dtype=weight_dtype)
    section_anchor_latents = materialized_batch["section_anchor_latents"].to(dtype=weight_dtype)
    x0_latents = materialized_batch["x0_latents"].to(dtype=weight_dtype)
    valid_target_frames = materialized_batch["valid_target_frames"]

    # 这段代码讲short,mid,long三档历史和target一起拼成 transformer 输入，后续 transformer 内部会区分处理。
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
    noisy_model_input = overwrite_anchor_latents(noisy_model_input, section_anchor_latents)
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
            latents_current_low_target=current_low_target_latents,
            return_dict=False,
        )[0]

    # 序列尾部可能被 padding，因此 loss 只统计真实存在的 target 帧。
    mask = build_valid_mask(
        valid_target_frames=valid_target_frames,
        latent_window_size=latent_window_size,
        device=device,
        dtype=torch.float32,
        valid_start=anchor_span_latents,
    )
    tail_position_weights = build_tail_position_weights(
        latent_window_size=latent_window_size,
        anchor_span_latents=anchor_span_latents,
        device=device,
        dtype=torch.float32,
    )
    weighted_mask = mask * tail_position_weights
    sigma_view_float = sigma.view(-1, 1, 1, 1, 1)
    xt = noisy_model_input.float()
    x_target = model_input.float()
    noise_target = noise.float()
    flow_target = flow_target.float()
    flow_pred = flow_pred.float()
    x_pred = xt - sigma_view_float * flow_pred
    noise_pred = xt + (1.0 - sigma_view_float) * flow_pred
    x_pred_for_temporal = overwrite_anchor_latents(x_pred.clone(), section_anchor_latents.float())
    x_target_for_temporal = overwrite_anchor_latents(x_target.clone(), section_anchor_latents.float())

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
    flow_loss = masked_mean(flow_sq_error * weighting, weighted_mask)
    x_loss = normalized_masked_mse(x_pred, x_target, weighted_mask)
    noise_loss = normalized_masked_mse(noise_pred, noise_target, weighted_mask)
    temporal_delta_loss = compute_temporal_delta_loss(
        prediction=x_pred_for_temporal,
        target=x_target_for_temporal,
        valid_target_frames=valid_target_frames,
        device=device,
        position_weights=tail_position_weights,
    )
    loss = (
        flow_loss_weight * flow_loss
        + aux_scale
        * (
            x_loss_weight * x_loss
            + noise_loss_weight * noise_loss
            + temporal_delta_loss_weight * temporal_delta_loss
        )
    )
    return TrainingStepOutput(
        loss=loss,
        raw_loss=raw_loss,
        flow_loss=flow_loss,
        x_loss=x_loss,
        noise_loss=noise_loss,
        temporal_delta_loss=temporal_delta_loss,
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


def build_tail_position_weights(
    latent_window_size: int,
    anchor_span_latents: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.zeros(1, 1, latent_window_size, 1, 1, device=device, dtype=dtype)
    tail_frames = max(0, latent_window_size - anchor_span_latents)
    if tail_frames <= 0:
        return weights

    if tail_frames == 1:
        weights[:, :, anchor_span_latents:, :, :] = 1.0
        return weights

    tail_positions = torch.arange(tail_frames, device=device, dtype=dtype)
    tail_weights = 1.0 + tail_positions / float(tail_frames - 1)
    weights[:, :, anchor_span_latents:, 0, 0] = tail_weights
    return weights


def overwrite_anchor_latents(latents: torch.Tensor, anchor_latents: torch.Tensor) -> torch.Tensor:
    anchor_steps = int(anchor_latents.shape[2])
    if anchor_steps <= 0:
        return latents
    latents[:, :, :anchor_steps] = anchor_latents.to(device=latents.device, dtype=latents.dtype)
    return latents


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
    valid_start: int = 0,
) -> torch.Tensor:
    """为每个样本构造目标窗口的有效帧掩码。"""
    batch_size = valid_target_frames.shape[0]
    mask = torch.zeros(batch_size, 1, latent_window_size, 1, 1, device=device, dtype=dtype)
    for idx, valid in enumerate(valid_target_frames.tolist()):
        mask[idx, :, valid_start : int(valid)] = 1
    return mask


def compute_temporal_delta_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_target_frames: torch.Tensor,
    device: torch.device,
    position_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if prediction.shape[2] <= 1:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    delta_prediction = prediction[:, :, 1:] - prediction[:, :, :-1]
    delta_target = target[:, :, 1:] - target[:, :, :-1]
    delta_mask = build_valid_mask(
        valid_target_frames=torch.clamp(valid_target_frames - 1, min=0),
        latent_window_size=prediction.shape[2] - 1,
        device=device,
        dtype=torch.float32,
        valid_start=0,
    )
    if position_weights is not None:
        delta_mask = delta_mask * position_weights[:, :, 1:, :, :]
    return masked_mean((delta_prediction - delta_target).abs(), delta_mask)


def get_scheduler_config_value(scheduler: object, key: str, default=None):
    """兼容 ConfigMixin / dict 风格读取 scheduler 配置。"""
    config = getattr(scheduler, "config", None)
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def get_model_cache_context(model: torch.nn.Module, cache_key: str):
    """优先复用模型自带 cache_context，缺失时退化为 no-op。"""
    cache_context = getattr(model, "cache_context", None)
    if callable(cache_context):
        return cache_context(cache_key)
    return nullcontext()


def model_supports_argument(model: torch.nn.Module, argument_name: str) -> bool:
    """检查模型 forward 是否支持某个关键字参数。"""
    try:
        signature = inspect.signature(model.forward)
    except (AttributeError, TypeError, ValueError):
        return False

    if argument_name in signature.parameters:
        return True
    return any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())


def tensor_abs_max(tensor: torch.Tensor) -> float:
    """计算 tensor 的绝对值最大值，NaN 会被保留为可诊断的 inf 级异常。"""
    if tensor.numel() == 0:
        return 0.0
    safe_tensor = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=float("inf"), neginf=float("-inf"))
    return float(safe_tensor.abs().max().item())


def ensure_finite_recover_state(
    *,
    stage: str,
    input_path: Path,
    section_start: Optional[int],
    step_idx: Optional[int],
    timestep: Optional[torch.Tensor | float],
    sigma: Optional[float],
    latents: torch.Tensor,
    noise_pred: Optional[torch.Tensor] = None,
) -> None:
    """在 recover 推理中对关键 tensor 做 fail-fast 非有限值检查。"""
    latents_finite = bool(torch.isfinite(latents).all().item())
    noise_pred_finite = True if noise_pred is None else bool(torch.isfinite(noise_pred).all().item())
    if latents_finite and noise_pred_finite:
        return

    if torch.is_tensor(timestep):
        timestep_value = float(timestep.detach().float().item())
    elif timestep is None:
        timestep_value = float("nan")
    else:
        timestep_value = float(timestep)

    sigma_value = float("nan") if sigma is None else float(sigma)
    noise_pred_abs_max = "n/a" if noise_pred is None else f"{tensor_abs_max(noise_pred):.6f}"
    raise RuntimeError(
        "Non-finite recover tensor detected "
        f"stage={stage} input_path={input_path} "
        f"section_start={'n/a' if section_start is None else section_start} "
        f"step_idx={'n/a' if step_idx is None else step_idx} "
        f"timestep={timestep_value:.6f} sigma={sigma_value:.6f} "
        f"latents_finite={latents_finite} latents_abs_max={tensor_abs_max(latents):.6f} "
        f"noise_pred_finite={noise_pred_finite} noise_pred_abs_max={noise_pred_abs_max}"
    )


def get_scheduler_sigma_for_step(scheduler: object, step_idx: int) -> Optional[float]:
    """读取当前 denoise step 对应的 sigma，便于错误日志定位。"""
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(sigmas, torch.Tensor) or step_idx < 0 or step_idx >= sigmas.shape[0]:
        return None
    return float(sigmas[step_idx].detach().float().item())


def validate_stage1_inference_scheduler(scheduler: object) -> None:
    """校验 recover 推理阶段的 scheduler 是否处于数值安全状态。"""
    timesteps = getattr(scheduler, "timesteps", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if not isinstance(timesteps, torch.Tensor) or not isinstance(sigmas, torch.Tensor):
        raise RuntimeError("Recover scheduler must expose tensor timesteps and sigmas.")
    if timesteps.ndim != 1 or sigmas.ndim != 1:
        raise RuntimeError(
            f"Recover scheduler expects 1D timesteps/sigmas, got timesteps={tuple(timesteps.shape)} "
            f"sigmas={tuple(sigmas.shape)}."
        )
    if sigmas.shape[0] != timesteps.shape[0] + 1:
        raise RuntimeError(
            f"Recover scheduler expects len(sigmas)=len(timesteps)+1, got "
            f"{sigmas.shape[0]} vs {timesteps.shape[0]}."
        )
    if not torch.isfinite(timesteps).all():
        raise RuntimeError("Recover scheduler timesteps contain non-finite values.")
    if not torch.isfinite(sigmas).all():
        raise RuntimeError("Recover scheduler sigmas contain non-finite values.")

    first_sigma = float(sigmas[0].detach().float().item())
    last_sigma = float(sigmas[-1].detach().float().item())
    if not (0.0 < first_sigma < 1.0):
        raise RuntimeError(
            f"Recover scheduler first sigma must satisfy 0 < sigma < 1 for stable stage1 inference, "
            f"got {first_sigma:.6f}."
        )
    if not math.isclose(last_sigma, 0.0, abs_tol=1e-8):
        raise RuntimeError(f"Recover scheduler last sigma must be 0, got {last_sigma:.6f}.")


def prepare_stage1_inference_scheduler(
    scheduler: object,
    latents: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    """把 recover 推理调度对齐到 Helios stage1 的时序设置。"""
    use_dynamic_shifting = bool(get_scheduler_config_value(scheduler, "use_dynamic_shifting", False))
    # 某些 Helios scheduler 实现在 `use_dynamic_shifting=True` 时会要求 `mu` 非空，
    # 即使我们随后会按 stage1 逻辑手动覆盖 timesteps/sigmas，这里也先给一个占位值，
    # 避免其内部初始化阶段直接因 `mu=None` 报错。
    if use_dynamic_shifting:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, mu=1.0)
    else:
        scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device)

    if use_dynamic_shifting:
        apply_schedule_shift = import_schedule_shift_fn()
        sigmas = torch.linspace(0.999, 0.0, steps=num_inference_steps + 1, dtype=torch.float32, device=device)[:-1]
        sigmas = apply_schedule_shift(
            sigmas=sigmas,
            noise=latents,
            base_seq_len=get_scheduler_config_value(scheduler, "base_image_seq_len", 256),
            max_seq_len=get_scheduler_config_value(scheduler, "max_image_seq_len", 4096),
            base_shift=get_scheduler_config_value(scheduler, "base_shift", 0.5),
            max_shift=get_scheduler_config_value(scheduler, "max_shift", 1.15),
        )
        num_train_timesteps = float(get_scheduler_config_value(scheduler, "num_train_timesteps", 1000))
        scheduler.timesteps = sigmas * num_train_timesteps
        scheduler.sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)])
        if hasattr(scheduler, "reset_scheduler_history"):
            scheduler.reset_scheduler_history()

    validate_stage1_inference_scheduler(scheduler)


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
    anchor_span_latents: int,
    num_inference_steps: int,
    seed: int,
    fix_anchor_during_denoise: bool,
    distributed_context: DistributedContext,
) -> torch.Tensor:
    """把一段 low latents 重建成 recover latents。

    这里按 section 分段恢复，每个 section 只依赖：
    - 首帧 `x0`
    - 低码率历史窗口
    - 当前 section 的完整 low window（anchor+tail）

    这与训练时的数据组织保持一致，也更贴近真实压缩恢复链路中的分段重建过程。
    """
    clean_full = sequence.clean_full_latents
    low_full = sequence.low_full_latents
    if clean_full.shape[1] == 1:
        return clean_full.clone().cpu().contiguous()

    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    history_sizes = list(history_sizes)
    section_starts = list(range(1, clean_full.shape[1], latent_window_size))
    recovered_sections: List[torch.Tensor] = []
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
        section_starts,
        desc=f"Reconstructing sections rank={distributed_context.rank}",
        disable=distributed_context.is_distributed and not distributed_context.is_main_process,
    ):
        history_latents = extract_history_window(
            low_full_latents=low_full,
            section_start=section_start,
            history_window_size=sum(history_sizes),
        ).unsqueeze(0)
        valid_target_frames = min(latent_window_size, clean_full.shape[1] - section_start)
        anchor_steps = min(anchor_span_latents, valid_target_frames)
        section_anchor_latents = low_full[:, section_start : section_start + anchor_steps].unsqueeze(0)
        current_low_target_latents, low_valid_frames = extract_current_low_target_window(
            low_full_latents=low_full,
            section_start=section_start,
            latent_window_size=latent_window_size,
        )
        if low_valid_frames != valid_target_frames:
            raise RuntimeError(
                f"Current low target valid frames mismatch at section_start={section_start}: "
                f"clean={valid_target_frames}, low={low_valid_frames}"
            )

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

        section_latents = run_stage1_denoise(
            transformer=transformer,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            device=device,
            weight_dtype=weight_dtype,
            input_path=sequence.path,
            section_start=section_start,
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
            anchor_latents=section_anchor_latents.to(device=device, dtype=weight_dtype),
            current_low_target_latents=current_low_target_latents.unsqueeze(0).to(device=device, dtype=weight_dtype),
            fix_anchor_during_denoise=fix_anchor_during_denoise,
        )
        valid_section = section_latents[0, :, :valid_target_frames].contiguous()
        recovered_sections.append(valid_section)

    recovered_remainder = torch.cat(recovered_sections, dim=1)
    recovered_remainder = recovered_remainder[:, : clean_full.shape[1] - 1]
    return torch.cat([clean_full[:, :1].cpu(), recovered_remainder], dim=1).contiguous()


@torch.inference_mode()
def run_stage1_denoise(
    transformer: torch.nn.Module,
    scheduler: object,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    input_path: Path,
    section_start: int,
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
    anchor_latents: torch.Tensor,
    current_low_target_latents: torch.Tensor,
    fix_anchor_during_denoise: bool,
) -> torch.Tensor:
    """执行单个 section 的扩散式去噪恢复。

    当前 section 的完整 low window 会作为额外条件分支输入 transformer，
    但扩散主体仍保持“从噪声开始恢复 clean target”的流程，不直接把 low tail 当作目标或初始化结果。
    """
    base_transformer = unwrap_model(transformer)
    generator = torch.Generator(device=device).manual_seed(seed)
    # 输入 hidden_states 是当前 section 的待恢复 latent，先随机初始化为高斯噪声；当前配置下 shape 近似是 [1, 16, 9, H, W]
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32, generator=generator)
    if fix_anchor_during_denoise:
        latents = overwrite_anchor_latents(latents, anchor_latents.float())
    prepare_stage1_inference_scheduler(
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    scheduler_type = str(get_scheduler_config_value(scheduler, "scheduler_type", "")).lower()
    supports_first_step_flag = model_supports_argument(base_transformer, "is_first_denoising_step")

    for step_idx, timestep in enumerate(scheduler.timesteps):
        if fix_anchor_during_denoise:
            latents = overwrite_anchor_latents(latents, anchor_latents.float())
        current_sigma = get_scheduler_sigma_for_step(scheduler, step_idx)
        ensure_finite_recover_state(
            stage="pre_model",
            input_path=input_path,
            section_start=section_start,
            step_idx=step_idx,
            timestep=timestep,
            sigma=current_sigma,
            latents=latents,
        )
        timestep_batch = timestep.expand(latents.shape[0])
        latent_model_input = latents.to(weight_dtype)
        transformer_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timestep_batch,
            "encoder_hidden_states": prompt_embeds,
            "indices_hidden_states": indices_hidden_states,
            "indices_latents_history_short": indices_latents_history_short,
            "indices_latents_history_mid": indices_latents_history_mid,
            "indices_latents_history_long": indices_latents_history_long,
            "latents_history_short": latents_history_short.to(weight_dtype),
            "latents_history_mid": latents_history_mid.to(weight_dtype),
            "latents_history_long": latents_history_long.to(weight_dtype),
            "latents_current_low_target": current_low_target_latents.to(weight_dtype),
            "return_dict": False,
        }
        if supports_first_step_flag:
            transformer_kwargs["is_first_denoising_step"] = step_idx == 0

        with get_model_cache_context(base_transformer, "cond"):
            # DiT的前向过程
            noise_pred = transformer(
                **transformer_kwargs,
            )[0]
        ensure_finite_recover_state(
            stage="post_model",
            input_path=input_path,
            section_start=section_start,
            step_idx=step_idx,
            timestep=timestep,
            sigma=current_sigma,
            latents=latents,
            noise_pred=noise_pred,
        )
        #  scheduler 用DiT前向的输出更新 latents，进入下一轮
        if scheduler_type == "unipc" and hasattr(scheduler, "step_unipc"):
            latents = scheduler.step_unipc(noise_pred.float(), timestep, latents, return_dict=False)[0]
        else:
            latents = scheduler.step(noise_pred.float(), timestep, latents, return_dict=False)[0]
        if fix_anchor_during_denoise:
            latents = overwrite_anchor_latents(latents, anchor_latents.float())
        ensure_finite_recover_state(
            stage="post_step",
            input_path=input_path,
            section_start=section_start,
            step_idx=step_idx,
            timestep=timestep,
            sigma=current_sigma,
            latents=latents,
            noise_pred=noise_pred,
        )

    if hasattr(base_transformer, "clear_kv_cache"):
        # 长序列逐段推理后清 cache，避免显存逐步累积。
        base_transformer.clear_kv_cache()
    return latents.cpu()


def build_low_latent_payload(sequence: PreparedSequence) -> Dict[str, object]:
    """构造 low_latents 的落盘 payload。"""
    if sequence.low_codec_payload is None:
        raise ValueError("sequence.low_codec_payload is missing. Please materialize low latents first.")
    return {
        "format_version": sequence.low_codec_payload["format_version"],
        "input_path": str(sequence.path),
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
        "chunk_lengths": list(sequence.metadata.chunk_lengths),
        "codec_config_v2": sequence.low_codec_payload["codec_config"],
        "codec_config": sequence.low_codec_payload["codec_config"],
        "tail_codec_type": sequence.low_codec_payload["codec_config"].get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE),
        "global_keyframe": sequence.low_codec_payload["global_keyframe"],
        "section_ranges": [list(section_range) for section_range in sequence.low_codec_payload["section_ranges"]],
        "section_anchor_payloads": sequence.low_codec_payload["section_anchor_payloads"],
        "section_tail_payloads": sequence.low_codec_payload["section_tail_payloads"],
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
            "section_ranges": [list(section_range) for section_range in sequence.low_codec_payload["section_ranges"]],
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


def compute_anchor_tail_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent_window_size: int,
    anchor_span_latents: int,
    mode: str,
) -> Dict[str, float]:
    if mode not in {"anchor", "tail"}:
        raise ValueError(f"mode must be 'anchor' or 'tail', got {mode}.")

    time_mask = torch.zeros(prediction.shape[1], dtype=torch.bool)
    for section_start, section_end in make_section_ranges(
        total_latent_frames=prediction.shape[1],
        section_span_latents=latent_window_size,
        start_index=1,
    ):
        anchor_end = min(section_end, section_start + anchor_span_latents)
        if mode == "anchor":
            time_mask[section_start:anchor_end] = True
        else:
            time_mask[anchor_end:section_end] = True

    if not bool(time_mask.any().item()):
        return {"mse": 0.0, "l1": 0.0, "psnr": float("inf")}

    prediction_selected = prediction[:, time_mask]
    target_selected = target[:, time_mask]
    return compute_tensor_metrics(prediction_selected, target_selected)


def compute_tail_position_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent_window_size: int,
    anchor_span_latents: int,
) -> Dict[str, Dict[str, float]]:
    tail_metrics: Dict[str, Dict[str, float]] = {}
    tail_span = max(0, latent_window_size - anchor_span_latents)
    total_latent_frames = prediction.shape[1]

    for tail_position in range(tail_span):
        selected_indices: List[int] = []
        for section_start, section_end in make_section_ranges(
            total_latent_frames=total_latent_frames,
            section_span_latents=latent_window_size,
            start_index=1,
        ):
            frame_index = section_start + anchor_span_latents + tail_position
            if frame_index < section_end:
                selected_indices.append(frame_index)

        key = f"position_{tail_position}"
        if not selected_indices:
            tail_metrics[key] = {"mse": 0.0, "l1": 0.0, "psnr": float("inf"), "count": 0}
            continue

        prediction_selected = prediction[:, selected_indices]
        target_selected = target[:, selected_indices]
        tail_metrics[key] = compute_tensor_metrics(prediction_selected, target_selected)
        tail_metrics[key]["count"] = int(len(selected_indices))
    return tail_metrics


def compute_temporal_delta_l1(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape[1] <= 1:
        return 0.0
    pred_delta = prediction[:, 1:] - prediction[:, :-1]
    target_delta = target[:, 1:] - target[:, :-1]
    return float((pred_delta - target_delta).abs().mean().item())


def compute_temporal_backtrack_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    total_latent_frames = prediction.shape[1]
    if total_latent_frames <= 1:
        return {
            "max_temporal_backtrack_latents": 0.0,
            "mean_temporal_backtrack_latents": 0.0,
        }

    prediction_flat = prediction.permute(1, 0, 2, 3).reshape(total_latent_frames, -1).float()
    target_flat = target.permute(1, 0, 2, 3).reshape(total_latent_frames, -1).float()
    nearest_indices: List[torch.Tensor] = []
    for start in range(0, total_latent_frames, 16):
        distances = torch.cdist(prediction_flat[start : start + 16], target_flat)
        nearest_indices.append(distances.argmin(dim=1).cpu())

    nearest_clean_indices = torch.cat(nearest_indices, dim=0).to(dtype=torch.float32)
    current_indices = torch.arange(total_latent_frames, dtype=torch.float32)
    absolute_backtrack = (nearest_clean_indices - current_indices).abs()
    return {
        "max_temporal_backtrack_latents": int(absolute_backtrack.max().item()),
        "mean_temporal_backtrack_latents": float(absolute_backtrack.mean().item()),
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
    learned_tail_codec: Optional[torch.nn.Module],
    training_model: Optional[torch.nn.Module],
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
    is_main_process = accelerator is None or accelerator.is_main_process
    staged_output_dir = output_dir
    temp_output_dir = None

    if is_main_process:
        if is_managed_checkpoint_dir(output_dir):
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            temp_output_dir = build_temporary_checkpoint_dir(output_dir)
            temp_output_dir.mkdir(parents=True, exist_ok=False)
            staged_output_dir = temp_output_dir
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    try:
        bundled_state_dict = None
        if training_model is not None:
            bundled_state_dict = get_cpu_state_dict(training_model, accelerator=accelerator)
        if bundled_state_dict is not None:
            if is_main_process:
                torch_save_atomic(
                    extract_prefixed_state_dict(bundled_state_dict, "transformer"),
                    staged_output_dir / "transformer_full.pt",
                )
                learned_state_dict = extract_prefixed_state_dict(bundled_state_dict, "learned_tail_codec")
                if learned_tail_codec is not None:
                    torch_save_atomic(learned_state_dict, staged_output_dir / LEARNED_CODEC_CHECKPOINT_NAME)
        else:
            save_transformer_checkpoint(transformer, staged_output_dir / "transformer_full.pt", accelerator=accelerator)
            save_optional_module_checkpoint(
                module=learned_tail_codec,
                output_path=staged_output_dir / LEARNED_CODEC_CHECKPOINT_NAME,
                accelerator=accelerator,
            )

        recover_config = build_recover_config(
            codec_config=codec_config,
            history_sizes=history_sizes,
            latent_window_size=latent_window_size,
            args=args,
            learned_tail_codec=learned_tail_codec,
        )
        if is_main_process:
            write_json_atomic(staged_output_dir / "recover_config.json", recover_config)
            write_json_atomic(staged_output_dir / "train_metrics.json", train_metrics)

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if is_main_process and temp_output_dir is not None:
            publish_checkpoint_directory(temp_output_dir=temp_output_dir, output_dir=output_dir)
            publish_latest_checkpoint_pointer(pointer_dir=output_dir.parent, checkpoint_dir=output_dir)
        elif is_main_process:
            publish_latest_checkpoint_pointer(pointer_dir=output_dir / "checkpoints", checkpoint_dir=output_dir)
    except Exception:
        if is_main_process and temp_output_dir is not None and temp_output_dir.exists():
            shutil.rmtree(temp_output_dir, ignore_errors=True)
        raise

    if accelerator is not None:
        accelerator.wait_for_everyone()


def build_recover_config(
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    args: argparse.Namespace,
    learned_tail_codec: Optional[torch.nn.Module],
) -> Dict[str, object]:
    """构造推理所需的最小 recover 配置。"""
    learned_codec_config = None
    if learned_tail_codec is not None:
        learned_codec_config = unwrap_model(learned_tail_codec).codec_hyperparams()
    return {
        "format_version": RECOVER_CONFIG_VERSION,
        "base_model_path": args.base_model_path,
        "weight_dtype": args.weight_dtype,
        "history_sizes": list(history_sizes),
        "latent_window_size": latent_window_size,
        "section_span_latents": codec_config.section_span_latents,
        "anchor_span_latents": codec_config.anchor_span_latents,
        "tail_span_latents": codec_config.tail_span_latents,
        "anchor_quant_dtype": codec_config.anchor_quant_dtype,
        "anchor_spatial_factor": codec_config.anchor_spatial_factor,
        "tail_codec_type": codec_config.tail_codec_type,
        "low_latent_format_version": (
            DEFAULT_LOW_LATENT_FORMAT_VERSION
            if codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE
            else LOW_LATENT_FORMAT_V2
        ),
        "learned_codec_checkpoint": LEARNED_CODEC_CHECKPOINT_NAME if learned_tail_codec is not None else None,
        "learned_codec_config": learned_codec_config,
        "use_current_low_target_branch": True,
        "current_low_target_mode": CURRENT_LOW_TARGET_MODE_FULL_WINDOW_1X,
        "current_low_target_init": CURRENT_LOW_TARGET_INIT_PATCH_SHORT,
        "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
        "codec_config": asdict(codec_config),
        "train_loss_config": {
            "loss_weighting_scheme": args.loss_weighting_scheme,
            "logit_mean": args.logit_mean,
            "logit_std": args.logit_std,
            "mode_scale": args.mode_scale,
            "flow_loss_weight": args.flow_loss_weight,
            "x_loss_weight": args.x_loss_weight,
            "noise_loss_weight": args.noise_loss_weight,
            "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
            "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
            "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        },
    }


def save_transformer_checkpoint(transformer: torch.nn.Module, output_path: Path, accelerator=None) -> None:
    """把 transformer 权重转到 CPU 后保存，减少 checkpoint 与设备环境的耦合。"""
    save_module_checkpoint(transformer, output_path, accelerator=accelerator)


def save_optional_module_checkpoint(
    module: Optional[torch.nn.Module],
    output_path: Path,
    accelerator=None,
) -> None:
    if module is None:
        return
    save_module_checkpoint(module, output_path, accelerator=accelerator)


def save_module_checkpoint(module: torch.nn.Module, output_path: Path, accelerator=None) -> None:
    cpu_state_dict = get_cpu_state_dict(module, accelerator=accelerator)
    if cpu_state_dict is None:
        return
    torch_save_atomic(cpu_state_dict, output_path)


def torch_save_atomic(payload: object, output_path: Path) -> None:
    """以原子替换方式保存大型 checkpoint 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = build_atomic_temp_path(output_path)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_json_atomic(output_path: Path, payload: object) -> None:
    """原子写入 JSON，避免推理读到半写入配置。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = build_atomic_temp_path(output_path)
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_text_atomic(output_path: Path, content: str) -> None:
    """原子写入文本元信息文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = build_atomic_temp_path(output_path)
    try:
        temp_path.write_text(content, encoding="utf-8")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def publish_checkpoint_directory(temp_output_dir: Path, output_dir: Path) -> None:
    """把 staging checkpoint 原子发布为正式 checkpoint 目录。"""
    backup_dir = None
    if output_dir.exists():
        backup_dir = output_dir.parent / f".{output_dir.name}.backup_{os.getpid()}_{time.time_ns()}"
        os.replace(output_dir, backup_dir)

    try:
        os.replace(temp_output_dir, output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            os.replace(backup_dir, output_dir)
        raise

    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


def publish_latest_checkpoint_pointer(pointer_dir: Path, checkpoint_dir: Path) -> None:
    """更新最近一次完整可推理 checkpoint 的稳定入口。"""
    pointer_dir.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(checkpoint_dir, start=pointer_dir)
    write_text_atomic(pointer_dir / LATEST_CHECKPOINT_POINTER_NAME, f"{relative_target}\n")

    temp_link = pointer_dir / f".{LATEST_CHECKPOINT_LINK_NAME}.tmp_{os.getpid()}_{time.time_ns()}"
    link_path = pointer_dir / LATEST_CHECKPOINT_LINK_NAME
    try:
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()
        os.symlink(relative_target, temp_link)
        os.replace(temp_link, link_path)
    except OSError:
        if temp_link.exists() or temp_link.is_symlink():
            temp_link.unlink()


def resolve_accelerator_state_dict_target(module: torch.nn.Module, accelerator) -> Optional[torch.nn.Module]:
    """仅对真正交给 accelerator.prepare 的模块走 accelerator.get_state_dict。

    当前 recover 训练里：
    - `transformer` 会被 Accelerate / DeepSpeed 接管
    - `learned_tail_codec` 保持各 rank 的本地副本，并手动同步梯度

    因此保存 checkpoint 时需要区分这两类模块，避免把普通 `nn.Module`
    误当成 DeepSpeedEngine 调用专属接口。
    """
    prepared_models = getattr(accelerator, "_models", None)
    if not prepared_models:
        return None

    base_module = unwrap_model(module)
    for prepared_model in prepared_models:
        if prepared_model is module:
            return prepared_model
        if unwrap_model(prepared_model) is base_module:
            return prepared_model
    return None


def get_cpu_state_dict(module: torch.nn.Module, accelerator=None) -> Optional[Dict[str, torch.Tensor]]:
    if accelerator is not None:
        state_dict_target = resolve_accelerator_state_dict_target(module, accelerator)
        if state_dict_target is not None:
            state_dict = accelerator.get_state_dict(state_dict_target)
            if not accelerator.is_main_process:
                return None
        else:
            if not accelerator.is_main_process:
                return None
            state_dict = unwrap_model(module).state_dict()
    else:
        state_dict = unwrap_model(module).state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return cpu_state_dict


def extract_prefixed_state_dict(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix_token = f"{prefix}."
    extracted: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix_token):
            extracted[key[len(prefix_token) :]] = value
    return extracted


def load_recover_config(checkpoint_dir: Path) -> Dict[str, object]:
    """读取并校验 recover checkpoint 的配置文件。"""
    config_path = checkpoint_dir / "recover_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing recover config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_keys = {
        "format_version",
        "history_sizes",
        "latent_window_size",
        "codec_config",
        "use_current_low_target_branch",
        "current_low_target_mode",
        "current_low_target_init",
    }
    missing_keys = sorted(required_keys - config.keys())
    if missing_keys:
        raise KeyError(f"Recover config {config_path} is missing keys: {missing_keys}")

    format_version = config["format_version"]
    if format_version != RECOVER_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported recover config format in {config_path}: {format_version}. "
            f"This repo version only supports {RECOVER_CONFIG_VERSION}, because recover inference now requires "
            "the current low target condition branch."
        )
    if not bool(config["use_current_low_target_branch"]):
        raise ValueError(
            f"Recover config {config_path} has use_current_low_target_branch={config['use_current_low_target_branch']}, "
            "but the current repo requires this branch to be enabled."
        )
    if str(config["current_low_target_mode"]) != CURRENT_LOW_TARGET_MODE_FULL_WINDOW_1X:
        raise ValueError(
            f"Recover config {config_path} has unsupported current_low_target_mode={config['current_low_target_mode']}. "
            f"Expected {CURRENT_LOW_TARGET_MODE_FULL_WINDOW_1X}."
        )
    if str(config["current_low_target_init"]) != CURRENT_LOW_TARGET_INIT_PATCH_SHORT:
        raise ValueError(
            f"Recover config {config_path} has unsupported current_low_target_init={config['current_low_target_init']}. "
            f"Expected {CURRENT_LOW_TARGET_INIT_PATCH_SHORT}."
        )

    codec_config = dict(config["codec_config"])
    latent_window_size = int(config["latent_window_size"])
    codec_config.setdefault("section_span_latents", int(config.get("section_span_latents", latent_window_size)))
    codec_config.setdefault("anchor_span_latents", int(config.get("anchor_span_latents", DEFAULT_ANCHOR_SPAN_LATENTS)))
    codec_config.setdefault(
        "tail_span_latents",
        int(config.get("tail_span_latents", codec_config["section_span_latents"] - codec_config["anchor_span_latents"])),
    )
    codec_config.setdefault("anchor_quant_dtype", config.get("anchor_quant_dtype", DEFAULT_ANCHOR_QUANT_DTYPE))
    codec_config.setdefault("anchor_spatial_factor", int(config.get("anchor_spatial_factor", DEFAULT_ANCHOR_SPATIAL_FACTOR)))
    codec_config.setdefault("tail_codec_type", config.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    learned_codec_config = dict(config.get("learned_codec_config", {}))
    if "hidden_channels" in learned_codec_config and "learned_codec_hidden_channels" not in codec_config:
        codec_config["learned_codec_hidden_channels"] = int(learned_codec_config["hidden_channels"])

    config["format_version"] = RECOVER_CONFIG_VERSION
    config["codec_config"] = codec_config
    config["section_span_latents"] = int(codec_config["section_span_latents"])
    config["anchor_span_latents"] = int(codec_config["anchor_span_latents"])
    config["tail_span_latents"] = int(codec_config["tail_span_latents"])
    config["anchor_quant_dtype"] = codec_config["anchor_quant_dtype"]
    config["anchor_spatial_factor"] = int(codec_config["anchor_spatial_factor"])
    config["tail_codec_type"] = str(codec_config["tail_codec_type"])
    config["low_latent_format_version"] = config.get(
        "low_latent_format_version",
        DEFAULT_LOW_LATENT_FORMAT_VERSION if config["tail_codec_type"] == LEARNED_TAIL_CODEC_TYPE else LOW_LATENT_FORMAT_V2,
    )
    config["learned_codec_checkpoint"] = config.get("learned_codec_checkpoint")
    config["learned_codec_config"] = learned_codec_config
    config["use_current_low_target_branch"] = True
    config["current_low_target_mode"] = CURRENT_LOW_TARGET_MODE_FULL_WINDOW_1X
    config["current_low_target_init"] = CURRENT_LOW_TARGET_INIT_PATCH_SHORT
    config.setdefault("fix_anchor_during_denoise", DEFAULT_FIX_ANCHOR_DURING_DENOISE)
    train_loss_config = dict(config.get("train_loss_config", {}))
    train_loss_config.setdefault("temporal_delta_loss_weight", DEFAULT_TEMPORAL_DELTA_LOSS_WEIGHT)
    config["temporal_delta_loss_weight"] = train_loss_config["temporal_delta_loss_weight"]
    config["train_loss_config"] = train_loss_config
    return config


def resolve_checkpoint_dir_for_inference(checkpoint_dir: Path) -> Path:
    """把用户传入的 checkpoint 路径解析为一套完整可推理的目录。"""
    if (checkpoint_dir / "recover_config.json").exists():
        return checkpoint_dir

    latest_link = checkpoint_dir / LATEST_CHECKPOINT_LINK_NAME
    if latest_link.exists():
        latest_target = latest_link.resolve()
        if (latest_target / "recover_config.json").exists():
            return latest_target

    latest_pointer = checkpoint_dir / LATEST_CHECKPOINT_POINTER_NAME
    if latest_pointer.exists():
        relative_target = latest_pointer.read_text(encoding="utf-8").strip()
        if relative_target:
            candidate = (checkpoint_dir / relative_target).resolve()
            if (candidate / "recover_config.json").exists():
                return candidate

    if checkpoint_dir.exists() and checkpoint_dir.is_dir():
        candidate_dirs = sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".") and (path / "recover_config.json").exists()
        )
        if candidate_dirs:
            return candidate_dirs[-1]
        if (checkpoint_dir / "transformer_full.pt").exists():
            raise FileNotFoundError(
                f"Checkpoint directory {checkpoint_dir} only has weights but is missing recover_config.json. "
                "This usually means the checkpoint was not fully published for inference."
            )

    raise FileNotFoundError(
        f"Could not resolve an infer-ready checkpoint from {checkpoint_dir}. "
        f"Expected either a directory containing recover_config.json, "
        f"a `{LATEST_CHECKPOINT_LINK_NAME}` / `{LATEST_CHECKPOINT_POINTER_NAME}` pointer, "
        "or at least one completed checkpoint subdirectory."
    )


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


def load_learned_tail_codec(
    checkpoint_dir: Optional[Path],
    codec_config: CodecConfig,
    latent_channels: int,
    device: torch.device,
) -> Optional[torch.nn.Module]:
    if codec_config.tail_codec_type != LEARNED_TAIL_CODEC_TYPE:
        return None

    learned_tail_codec = build_learned_tail_codec(codec_config=asdict(codec_config), in_channels=latent_channels).to(device)
    if checkpoint_dir is None:
        learned_tail_codec.train()
        return learned_tail_codec

    state_dict_path = checkpoint_dir / LEARNED_CODEC_CHECKPOINT_NAME
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Missing learned codec checkpoint: {state_dict_path}")
    state_dict = load_state_dict_file(state_dict_path)
    incompatible = learned_tail_codec.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected learned codec state dict mismatch while loading {state_dict_path}: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    learned_tail_codec.eval()
    learned_tail_codec.requires_grad_(False)
    return learned_tail_codec


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


def synchronize_module_parameters(module: Optional[torch.nn.Module]) -> None:
    """把本地小模块参数从 rank0 广播到所有 rank，保证初始化一致。"""
    if module is None or not dist.is_initialized():
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def synchronize_module_gradients(module: Optional[torch.nn.Module]) -> None:
    """手动平均本地小模块梯度，避免把它纳入 ZeRO-3 参数收集序列。"""
    if module is None or not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return

    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter.data)
        dist.all_reduce(parameter.grad.data, op=dist.ReduceOp.SUM)
        parameter.grad.data.div_(float(world_size))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """剥离 DDP / accelerate 外层包装，拿到真正的底层模型。"""
    unwrapped = model
    while hasattr(unwrapped, "module"):
        next_model = getattr(unwrapped, "module")
        if next_model is unwrapped:
            break
        unwrapped = next_model
    return unwrapped


def get_training_submodule(model: torch.nn.Module, name: str) -> Optional[torch.nn.Module]:
    """从训练 bundle 或其外层包装中取出子模块。"""
    submodule = getattr(model, name, None)
    if submodule is not None:
        return submodule
    return getattr(unwrap_model(model), name, None)


def set_seed(seed: int) -> None:
    """统一设置 Python / PyTorch / CUDA 随机种子。"""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
