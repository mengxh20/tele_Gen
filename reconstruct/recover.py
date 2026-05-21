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
from collections import OrderedDict
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
    DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
    DEFAULT_SINGLE_HEAD_CODEC_TYPE,
    DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
    DEFAULT_DUAL_HEAD_CODEC_TYPE,
    DEFAULT_BOUNDARY_JUMP_THRESHOLD,
    DEFAULT_CUT_DETECTION_THRESHOLD,
    DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD,
    DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR,
    DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
    DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
    DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
    DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
    DEFAULT_MAX_PREDICT_ONLY_GAP_SECTIONS,
    DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD,
    DUAL_HEAD_ANCHOR_CODEC_TYPE,
    DUAL_HEAD_P_DELTA_CODEC_TYPE,
    DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
    DEFAULT_DUAL_TAIL_CODEC_TYPE,
    DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
    DUAL_REFRESH_SECTION_MODE,
    DUAL_TAIL_ANCHOR_CODEC_TYPE,
    DUAL_TAIL_P_DELTA_CODEC_TYPE,
    INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE,
    PREDICT_ONLY_SECTION_MODE,
    RAW_GLOBAL_KEYFRAME_CODEC_TYPE,
    SINGLE_HEAD_ANCHOR_CODEC_TYPE,
    SINGLE_HEAD_SOFT_POOL_CODEC_TYPE,
    SINGLE_REFRESH_SECTION_MODE,
    TRILINEAR_TAIL_CODEC_TYPE,
    decode_low_latents_payload,
    encode_anchor_plus_tail_latents,
    estimate_anchor_plus_tail_codec_byte_breakdown,
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
    LOW_LATENT_FORMAT_V4,
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
DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE = RAW_GLOBAL_KEYFRAME_CODEC_TYPE
DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE = "int8"
DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR = 1
LOW_CODEC_BYTE_BREAKDOWN_KEYS = (
    "global_keyframe_bytes",
    "single_head_bytes",
    "dual_head_bytes",
    "dual_tail_presidual_bytes",
    "metadata_bytes",
)
DEFAULT_HISTORY_SIZES = [3, 1, 1]
DEFAULT_LATENT_WINDOW_SIZE = 3  # Helios的VAE中4帧视频压缩成一个latent时间步，所以其实一个latent对应4帧,最终对应原视频的关系是 (DEFAULT_ANCHOR_SPAN_LATENTS + (DEFAULT_LATENT_WINDOW_SIZE - 1) * 4 = 原视频帧数)
DEFAULT_SECTION_SPAN_LATENTS = DEFAULT_LATENT_WINDOW_SIZE
DEFAULT_ANCHOR_SPAN_LATENTS = 1
DEFAULT_TAIL_SPAN_LATENTS = DEFAULT_SECTION_SPAN_LATENTS - DEFAULT_ANCHOR_SPAN_LATENTS
DEFAULT_ANCHOR_QUANT_DTYPE = "int8"
DEFAULT_ANCHOR_SPATIAL_FACTOR = 1
DEFAULT_SINGLE_HEAD_CODEC_TYPE = DEFAULT_SINGLE_HEAD_CODEC_TYPE
DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR = DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR
DEFAULT_DUAL_HEAD_CODEC_TYPE = DEFAULT_DUAL_HEAD_CODEC_TYPE
DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR: Optional[int] = DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR
DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR = DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR
DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR = DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR
DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO = DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO
DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR = DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR
DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO = DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO
DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR: Optional[int] = None
DEFAULT_TAIL_CODEC_TYPE = TRILINEAR_TAIL_CODEC_TYPE
DEFAULT_MAX_PREDICT_ONLY_GAP = DEFAULT_MAX_PREDICT_ONLY_GAP_SECTIONS
DEFAULT_SINGLE_REFRESH_GAIN = DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD
DEFAULT_DUAL_REFRESH_GAIN = DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD
DEFAULT_BOUNDARY_JUMP = DEFAULT_BOUNDARY_JUMP_THRESHOLD
DEFAULT_CUT_DETECTION = DEFAULT_CUT_DETECTION_THRESHOLD
DEFAULT_TARGET_LOW_BPP: Optional[float] = None
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
DEFAULT_AUX_LOSS_CLIP_VALUE: Optional[float] = None
DEFAULT_AUX_LOSS_SIGMA_MIN: Optional[float] = None
DEFAULT_AUX_LOSS_SIGMA_MAX: Optional[float] = None
DEFAULT_TEMPORAL_DELTA_LOSS_WEIGHT = 0.1
DEFAULT_FIX_ANCHOR_DURING_DENOISE = True
DEFAULT_SOFT_TAIL_HINT_STRENGTH = 0.35
DEFAULT_SOFT_TAIL_HINT_STEP_FRACTION = 0.5
DEFAULT_LOW_LATENT_FORMAT_VERSION = LOW_LATENT_FORMAT_V4
RECOVER_CONFIG_VERSION = "helios_recover_v4"
LEARNED_CODEC_CHECKPOINT_NAME = "learned_codec.pt"
LATEST_CHECKPOINT_LINK_NAME = "latest"
LATEST_CHECKPOINT_POINTER_NAME = "latest_checkpoint.txt"
EPS = 1e-8
REQUIRED_BASE_MODEL_SUBDIRS = ("tokenizer", "text_encoder", "transformer", "scheduler")
VALID_SECTION_MODES = {
    PREDICT_ONLY_SECTION_MODE,
    SINGLE_REFRESH_SECTION_MODE,
    DUAL_REFRESH_SECTION_MODE,
}


@dataclass
class CodecConfig:
    """低码率 latent 编码配置。

    当前版本的 codec 以“稀疏 refresh”为核心：
    - `global_keyframe(x0)`：全局唯一高精度锚点
    - `predict-only`：当前 section 不发送额外信息，纯靠先验和历史外推
    - `single-refresh`：当前 section 只发送一个头部 refresh anchor
    - `dual-refresh`：当前 section 发送头尾两个 refresh anchor

    section 的模式由启发式分数决定，低码率 `low_full_latents` 则按 section 顺序代理解码，
    供 recover 阶段作为历史条件和硬约束使用。
    """

    temporal_factor: int = DEFAULT_TEMPORAL_FACTOR
    spatial_factor: int = DEFAULT_SPATIAL_FACTOR
    quant_dtype: str = DEFAULT_QUANT_DTYPE
    keyframe_dtype: str = DEFAULT_KEYFRAME_DTYPE
    global_keyframe_codec_type: str = DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE
    global_keyframe_quant_dtype: str = DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE
    global_keyframe_spatial_factor: int = DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR
    section_span_latents: int = DEFAULT_SECTION_SPAN_LATENTS
    anchor_span_latents: int = DEFAULT_ANCHOR_SPAN_LATENTS
    tail_span_latents: int = DEFAULT_TAIL_SPAN_LATENTS
    anchor_quant_dtype: str = DEFAULT_ANCHOR_QUANT_DTYPE
    anchor_spatial_factor: int = DEFAULT_ANCHOR_SPATIAL_FACTOR
    single_head_codec_type: str = DEFAULT_SINGLE_HEAD_CODEC_TYPE
    single_head_soft_pool_spatial_factor: int = DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR
    dual_head_codec_type: str = DEFAULT_DUAL_HEAD_CODEC_TYPE
    dual_head_anchor_spatial_factor: Optional[int] = DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR
    dual_head_p_delta_spatial_factor: int = DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR
    dual_head_soft_pool_spatial_factor: int = DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR
    adaptive_dual_head_full_ratio: float = DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO
    adaptive_dual_head_soft_pool_hard_factor: Optional[int] = DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR
    adaptive_dual_head_soft_pool_hard_ratio: float = DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO
    dual_tail_anchor_spatial_factor: Optional[int] = DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR
    dual_tail_codec_type: str = DEFAULT_DUAL_TAIL_CODEC_TYPE
    dual_tail_p_delta_spatial_factor: int = DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR
    tail_codec_type: str = DEFAULT_TAIL_CODEC_TYPE
    learned_codec_hidden_channels: Optional[int] = None
    max_predict_only_gap_sections: int = DEFAULT_MAX_PREDICT_ONLY_GAP
    single_refresh_gain_threshold: float = DEFAULT_SINGLE_REFRESH_GAIN
    dual_refresh_gain_threshold: float = DEFAULT_DUAL_REFRESH_GAIN
    boundary_jump_threshold: float = DEFAULT_BOUNDARY_JUMP
    cut_detection_threshold: float = DEFAULT_CUT_DETECTION
    target_low_bpp: Optional[float] = DEFAULT_TARGET_LOW_BPP


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
class LatentSequenceIndexEntry:
    """训练数据索引中的轻量条目，不持有 latent tensor。"""

    path: Path
    metadata: SequenceMetadata
    latent_channels: int
    total_latent_frames: int


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
        sequences: Sequence[LatentSequenceIndexEntry],
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
            total_latent_frames = sequence.total_latent_frames
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


class LazyPreparedSequenceStore:
    """按需加载训练 latent，并用小 LRU 缓存限制 CPU 常驻内存。

    旧训练流程会在每个 rank 启动时把所有 `.pt` 全部 `torch.load` 到 CPU；
    多卡时这份数据会按 rank 数重复，容易把宿主机内存顶满。这个 store 只在
    当前 batch 真正用到某个 `seq_idx` 时加载对应文件。
    """

    def __init__(
        self,
        entries: Sequence[LatentSequenceIndexEntry],
        codec_config: CodecConfig,
        cache_size: int,
    ):
        self.entries = list(entries)
        self.codec_config = codec_config
        self.cache_size = max(0, int(cache_size))
        self._cache: "OrderedDict[int, PreparedSequence]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, seq_idx: int) -> PreparedSequence:
        seq_idx = int(seq_idx)
        if seq_idx in self._cache:
            sequence = self._cache.pop(seq_idx)
            self._cache[seq_idx] = sequence
            return sequence

        sequence = prepare_sequence(
            self.entries[seq_idx].path,
            self.codec_config,
            learned_tail_codec=None,
            materialize_low_latents=False,
        )
        if self.cache_size > 0:
            self._cache[seq_idx] = sequence
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return sequence

    def clear(self) -> None:
        self._cache.clear()


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
    parser.add_argument(
        "--init_checkpoint_dir",
        type=Path,
        default=None,
        help="Optional recover checkpoint directory used to initialize the model for continued fine-tuning.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--checkpoint_every_steps",
        type=int,
        default=None,
        help="Save a periodic checkpoint every N optimizer steps. Disabled when unset.",
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--lazy_sequence_cache_size",
        type=int,
        default=2,
        help=(
            "Number of fully loaded latent sequences kept per rank during training. "
            "Use 0 to disable the CPU cache; low values avoid loading the whole dataset into RAM."
        ),
    )
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
    parser.add_argument(
        "--global_keyframe_codec_type",
        type=str,
        default=DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE,
        choices=[RAW_GLOBAL_KEYFRAME_CODEC_TYPE, INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE],
    )
    parser.add_argument(
        "--global_keyframe_quant_dtype",
        type=str,
        default=DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE,
        choices=["int8"],
    )
    parser.add_argument(
        "--global_keyframe_spatial_factor",
        type=int,
        default=DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR,
        help="Spatial downsample factor used when encoding the global keyframe payload.",
    )
    parser.add_argument("--section_span_latents", "--section", dest="section_span_latents", type=int, default=None)
    parser.add_argument(
        "--anchor_span_latents",
        "--anchor",
        dest="anchor_span_latents",
        type=int,
        default=DEFAULT_ANCHOR_SPAN_LATENTS,
    )
    parser.add_argument("--tail_span_latents", type=int, default=None)
    parser.add_argument("--anchor_quant_dtype", type=str, default=DEFAULT_ANCHOR_QUANT_DTYPE, choices=["int8"])
    parser.add_argument("--anchor_spatial_factor", type=int, default=DEFAULT_ANCHOR_SPATIAL_FACTOR)
    parser.add_argument(
        "--single_head_codec_type",
        type=str,
        default=DEFAULT_SINGLE_HEAD_CODEC_TYPE,
        choices=[SINGLE_HEAD_ANCHOR_CODEC_TYPE, SINGLE_HEAD_SOFT_POOL_CODEC_TYPE],
        help="Codec used for the head condition in single-refresh sections.",
    )
    parser.add_argument(
        "--single_head_soft_pool_spatial_factor",
        type=int,
        default=DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        help="Spatial downsample factor for soft pooled head payloads in single-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_codec_type",
        type=str,
        default=DEFAULT_DUAL_HEAD_CODEC_TYPE,
        choices=[DUAL_HEAD_ANCHOR_CODEC_TYPE, DUAL_HEAD_P_DELTA_CODEC_TYPE, DUAL_HEAD_SOFT_POOL_CODEC_TYPE],
        help="Codec used for the head condition in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_anchor_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR,
        help="Optional spatial downsample factor applied only to the head anchor in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_p_delta_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
        help="Spatial downsample factor for P-delta head payloads in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_soft_pool_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        help="Spatial downsample factor for soft pooled head payloads in dual-refresh sections.",
    )
    parser.add_argument(
        "--adaptive_dual_head_full_ratio",
        type=float,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
        help="Fraction of dual-refresh sections that keep full-resolution head anchors instead of the cheap dual-head variant.",
    )
    parser.add_argument(
        "--adaptive_dual_head_soft_pool_hard_factor",
        type=int,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
        help="Optional stronger soft-pool factor for the hardest dual-refresh head hints.",
    )
    parser.add_argument(
        "--adaptive_dual_head_soft_pool_hard_ratio",
        type=float,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
        help="Fraction of dual-refresh sections that use the stronger soft-pool head factor.",
    )
    parser.add_argument(
        "--dual_tail_anchor_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR,
        help="Optional spatial downsample factor applied only to the tail anchor in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_tail_codec_type",
        type=str,
        default=DEFAULT_DUAL_TAIL_CODEC_TYPE,
        choices=[DUAL_TAIL_ANCHOR_CODEC_TYPE, DUAL_TAIL_P_DELTA_CODEC_TYPE],
        help="Codec used for the tail condition in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_tail_p_delta_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
        help="Spatial downsample factor for P-delta tail payloads in dual-refresh sections.",
    )
    parser.add_argument("--max_predict_only_gap_sections", type=int, default=DEFAULT_MAX_PREDICT_ONLY_GAP)
    parser.add_argument("--single_refresh_gain_threshold", type=float, default=DEFAULT_SINGLE_REFRESH_GAIN)
    parser.add_argument("--dual_refresh_gain_threshold", type=float, default=DEFAULT_DUAL_REFRESH_GAIN)
    parser.add_argument("--boundary_jump_threshold", type=float, default=DEFAULT_BOUNDARY_JUMP)
    parser.add_argument("--cut_detection_threshold", type=float, default=DEFAULT_CUT_DETECTION)
    parser.add_argument(
        "--target_low_bpp",
        type=float,
        default=DEFAULT_TARGET_LOW_BPP,
        help="Optional low-bpp target used by budget-driven sparse refresh allocation.",
    )
    parser.add_argument(
        "--train_mixed_dual_tail_spatial_factor",
        type=int,
        default=None,
        help="Optional cheap-tail spatial factor used only during training for dual-refresh tail augmentation.",
    )
    parser.add_argument(
        "--train_mixed_dual_tail_ratio",
        type=float,
        default=0.0,
        help="Probability of replacing a dual-refresh tail anchor with the cheap-tail variant during training.",
    )
    parser.add_argument(
        "--tail_codec_type",
        type=str,
        default=DEFAULT_TAIL_CODEC_TYPE,
        choices=[TRILINEAR_TAIL_CODEC_TYPE, LEARNED_TAIL_CODEC_TYPE],
    )
    parser.add_argument("--history_sizes", type=int, nargs="+", default=DEFAULT_HISTORY_SIZES)
    parser.add_argument("--latent_window_size", type=int, default=DEFAULT_LATENT_WINDOW_SIZE)
    parser.add_argument(
        "--num_inference_steps",
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
        help="Default denoising steps saved into the checkpoint config and run name.",
    )
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
        "--aux_loss_clip_value",
        type=float,
        default=DEFAULT_AUX_LOSS_CLIP_VALUE,
        help="Optional max value for each auxiliary loss term before it enters optimization.",
    )
    parser.add_argument(
        "--aux_loss_sigma_min",
        type=float,
        default=DEFAULT_AUX_LOSS_SIGMA_MIN,
        help="Only apply auxiliary losses when sampled sigma is at least this value.",
    )
    parser.add_argument(
        "--aux_loss_sigma_max",
        type=float,
        default=DEFAULT_AUX_LOSS_SIGMA_MAX,
        help="Only apply auxiliary losses when sampled sigma is at most this value.",
    )
    parser.add_argument(
        "--fix_anchor_during_denoise",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FIX_ANCHOR_DURING_DENOISE,
    )
    parser.add_argument(
        "--soft_tail_hint_strength",
        type=float,
        default=DEFAULT_SOFT_TAIL_HINT_STRENGTH,
        help="Soft blend strength used for dual-refresh tail hints during training and inference.",
    )
    parser.add_argument(
        "--soft_tail_hint_step_fraction",
        type=float,
        default=DEFAULT_SOFT_TAIL_HINT_STEP_FRACTION,
        help="Fraction of denoising steps that receive dual-refresh soft tail hints during inference.",
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
    parser.add_argument(
        "--num_inference_steps",
        "--steps",
        dest="num_inference_steps",
        type=int,
        default=DEFAULT_NUM_INFERENCE_STEPS,
    )
    parser.add_argument(
        "--global_keyframe_codec_type",
        type=str,
        default=DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE,
        choices=[RAW_GLOBAL_KEYFRAME_CODEC_TYPE, INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE],
    )
    parser.add_argument(
        "--global_keyframe_quant_dtype",
        type=str,
        default=DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE,
        choices=["int8"],
    )
    parser.add_argument(
        "--global_keyframe_spatial_factor",
        type=int,
        default=DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR,
        help="Spatial downsample factor used when encoding the global keyframe payload.",
    )
    parser.add_argument(
        "--single_head_codec_type",
        type=str,
        default=DEFAULT_SINGLE_HEAD_CODEC_TYPE,
        choices=[SINGLE_HEAD_ANCHOR_CODEC_TYPE, SINGLE_HEAD_SOFT_POOL_CODEC_TYPE],
        help="Codec used for the head condition in single-refresh sections.",
    )
    parser.add_argument(
        "--single_head_soft_pool_spatial_factor",
        type=int,
        default=DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        help="Spatial downsample factor for soft pooled head payloads in single-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_codec_type",
        type=str,
        default=DEFAULT_DUAL_HEAD_CODEC_TYPE,
        choices=[DUAL_HEAD_ANCHOR_CODEC_TYPE, DUAL_HEAD_P_DELTA_CODEC_TYPE, DUAL_HEAD_SOFT_POOL_CODEC_TYPE],
        help="Codec used for the head condition in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_anchor_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR,
        help="Optional spatial downsample factor applied only to the head anchor in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_p_delta_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
        help="Spatial downsample factor for P-delta head payloads in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_head_soft_pool_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        help="Spatial downsample factor for soft pooled head payloads in dual-refresh sections.",
    )
    parser.add_argument(
        "--adaptive_dual_head_full_ratio",
        type=float,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
        help="Fraction of dual-refresh sections that keep full-resolution head anchors instead of the cheap dual-head variant.",
    )
    parser.add_argument(
        "--adaptive_dual_head_soft_pool_hard_factor",
        type=int,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
        help="Optional stronger soft-pool factor for the hardest dual-refresh head hints.",
    )
    parser.add_argument(
        "--adaptive_dual_head_soft_pool_hard_ratio",
        type=float,
        default=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
        help="Fraction of dual-refresh sections that use the stronger soft-pool head factor.",
    )
    parser.add_argument(
        "--dual_tail_anchor_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR,
        help="Optional spatial downsample factor applied only to the tail anchor in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_tail_codec_type",
        type=str,
        default=DEFAULT_DUAL_TAIL_CODEC_TYPE,
        choices=[DUAL_TAIL_ANCHOR_CODEC_TYPE, DUAL_TAIL_P_DELTA_CODEC_TYPE],
        help="Codec used for the tail condition in dual-refresh sections.",
    )
    parser.add_argument(
        "--dual_tail_p_delta_spatial_factor",
        type=int,
        default=DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
        help="Spatial downsample factor for P-delta tail payloads in dual-refresh sections.",
    )
    parser.add_argument(
        "--target_low_bpp",
        type=float,
        default=DEFAULT_TARGET_LOW_BPP,
        help="Optional low-bpp target used by budget-driven sparse refresh allocation.",
    )
    parser.add_argument("--max_predict_only_gap_sections", type=int, default=DEFAULT_MAX_PREDICT_ONLY_GAP)
    parser.add_argument("--single_refresh_gain_threshold", type=float, default=DEFAULT_SINGLE_REFRESH_GAIN)
    parser.add_argument("--dual_refresh_gain_threshold", type=float, default=DEFAULT_DUAL_REFRESH_GAIN)
    parser.add_argument("--boundary_jump_threshold", type=float, default=DEFAULT_BOUNDARY_JUMP)
    parser.add_argument("--cut_detection_threshold", type=float, default=DEFAULT_CUT_DETECTION)
    parser.add_argument(
        "--predict_only_steps",
        type=int,
        default=None,
        help="Override denoising steps for predict-only sections. Defaults to --steps when unset.",
    )
    parser.add_argument(
        "--single_refresh_steps",
        type=int,
        default=None,
        help="Override denoising steps for single-refresh sections. Defaults to --steps when unset.",
    )
    parser.add_argument(
        "--dual_refresh_steps",
        type=int,
        default=None,
        help="Override denoising steps for dual-refresh sections. Defaults to --steps when unset.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Only run inference on the first N latent files before distributed sharding.",
    )
    parser.add_argument(
        "--soft_tail_hint_strength",
        type=float,
        default=None,
        help="Override soft blend strength used for dual-refresh tail hints. Defaults to the checkpoint setting.",
    )
    parser.add_argument(
        "--soft_tail_hint_step_fraction",
        type=float,
        default=None,
        help="Override the fraction of denoising steps that receive dual-refresh soft tail hints.",
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
    if args.aux_loss_clip_value is not None and args.aux_loss_clip_value <= 0.0:
        raise ValueError(f"aux_loss_clip_value must be > 0 when set, got {args.aux_loss_clip_value}.")
    if args.aux_loss_sigma_min is not None and not 0.0 <= args.aux_loss_sigma_min <= 1.0:
        raise ValueError(f"aux_loss_sigma_min must be in [0, 1], got {args.aux_loss_sigma_min}.")
    if args.aux_loss_sigma_max is not None and not 0.0 <= args.aux_loss_sigma_max <= 1.0:
        raise ValueError(f"aux_loss_sigma_max must be in [0, 1], got {args.aux_loss_sigma_max}.")
    if (
        args.aux_loss_sigma_min is not None
        and args.aux_loss_sigma_max is not None
        and args.aux_loss_sigma_min > args.aux_loss_sigma_max
    ):
        raise ValueError(
            "aux_loss_sigma_min must be <= aux_loss_sigma_max, "
            f"got {args.aux_loss_sigma_min} > {args.aux_loss_sigma_max}."
        )
    if args.num_inference_steps <= 0:
        raise ValueError(f"num_inference_steps must be > 0, got {args.num_inference_steps}.")
    if args.checkpoint_every_steps is not None and int(args.checkpoint_every_steps) <= 0:
        raise ValueError(f"checkpoint_every_steps must be > 0 when set, got {args.checkpoint_every_steps}.")
    if int(getattr(args, "lazy_sequence_cache_size", 2)) < 0:
        raise ValueError(
            "lazy_sequence_cache_size must be >= 0, "
            f"got {getattr(args, 'lazy_sequence_cache_size')}."
        )
    target_low_bpp = getattr(args, "target_low_bpp", DEFAULT_TARGET_LOW_BPP)
    if target_low_bpp is not None and float(target_low_bpp) <= 0.0:
        raise ValueError(f"target_low_bpp must be > 0, got {target_low_bpp}.")
    train_mixed_dual_tail_ratio = float(getattr(args, "train_mixed_dual_tail_ratio", 0.0))
    if not 0.0 <= train_mixed_dual_tail_ratio <= 1.0:
        raise ValueError(
            f"train_mixed_dual_tail_ratio must be in [0, 1], got {train_mixed_dual_tail_ratio}."
        )
    train_mixed_dual_tail_spatial_factor = getattr(args, "train_mixed_dual_tail_spatial_factor", None)
    if train_mixed_dual_tail_spatial_factor is not None and int(train_mixed_dual_tail_spatial_factor) < 1:
        raise ValueError(
            "train_mixed_dual_tail_spatial_factor must be >= 1 when set, "
            f"got {train_mixed_dual_tail_spatial_factor}."
        )
    dual_tail_anchor_spatial_factor = getattr(
        args,
        "dual_tail_anchor_spatial_factor",
        DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR,
    )
    dual_head_anchor_spatial_factor = getattr(
        args,
        "dual_head_anchor_spatial_factor",
        DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR,
    )
    dual_head_codec_type = str(getattr(args, "dual_head_codec_type", DEFAULT_DUAL_HEAD_CODEC_TYPE))
    if dual_head_codec_type not in {
        DUAL_HEAD_ANCHOR_CODEC_TYPE,
        DUAL_HEAD_P_DELTA_CODEC_TYPE,
        DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
    }:
        raise ValueError(
            f"dual_head_codec_type must be one of "
            f"{DUAL_HEAD_ANCHOR_CODEC_TYPE}, {DUAL_HEAD_P_DELTA_CODEC_TYPE}, "
            f"{DUAL_HEAD_SOFT_POOL_CODEC_TYPE}; got {dual_head_codec_type}."
        )
    if dual_head_anchor_spatial_factor is not None and int(dual_head_anchor_spatial_factor) < 1:
        raise ValueError(
            "dual_head_anchor_spatial_factor must be >= 1 when set, "
            f"got {dual_head_anchor_spatial_factor}."
        )
    dual_head_p_delta_spatial_factor = int(
        getattr(args, "dual_head_p_delta_spatial_factor", DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR)
    )
    if dual_head_p_delta_spatial_factor < 1:
        raise ValueError(
            "dual_head_p_delta_spatial_factor must be >= 1, "
            f"got {dual_head_p_delta_spatial_factor}."
        )
    dual_head_soft_pool_spatial_factor = int(
        getattr(args, "dual_head_soft_pool_spatial_factor", DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR)
    )
    if dual_head_soft_pool_spatial_factor < 1:
        raise ValueError(
            "dual_head_soft_pool_spatial_factor must be >= 1, "
            f"got {dual_head_soft_pool_spatial_factor}."
        )
    adaptive_dual_head_full_ratio = float(
        getattr(args, "adaptive_dual_head_full_ratio", DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO)
    )
    if not 0.0 <= adaptive_dual_head_full_ratio <= 1.0:
        raise ValueError(
            "adaptive_dual_head_full_ratio must be in [0, 1], "
            f"got {adaptive_dual_head_full_ratio}."
        )
    adaptive_dual_head_soft_pool_hard_factor = getattr(
        args,
        "adaptive_dual_head_soft_pool_hard_factor",
        DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
    )
    if (
        adaptive_dual_head_soft_pool_hard_factor is not None
        and int(adaptive_dual_head_soft_pool_hard_factor) < 1
    ):
        raise ValueError(
            "adaptive_dual_head_soft_pool_hard_factor must be >= 1 when set, "
            f"got {adaptive_dual_head_soft_pool_hard_factor}."
        )
    adaptive_dual_head_soft_pool_hard_ratio = float(
        getattr(
            args,
            "adaptive_dual_head_soft_pool_hard_ratio",
            DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
        )
    )
    if not 0.0 <= adaptive_dual_head_soft_pool_hard_ratio <= 1.0:
        raise ValueError(
            "adaptive_dual_head_soft_pool_hard_ratio must be in [0, 1], "
            f"got {adaptive_dual_head_soft_pool_hard_ratio}."
        )
    if dual_tail_anchor_spatial_factor is not None and int(dual_tail_anchor_spatial_factor) < 1:
        raise ValueError(
            "dual_tail_anchor_spatial_factor must be >= 1 when set, "
            f"got {dual_tail_anchor_spatial_factor}."
        )
    soft_tail_hint_strength = float(getattr(args, "soft_tail_hint_strength", DEFAULT_SOFT_TAIL_HINT_STRENGTH))
    if soft_tail_hint_strength < 0.0:
        raise ValueError(f"soft_tail_hint_strength must be >= 0, got {soft_tail_hint_strength}.")
    soft_tail_hint_step_fraction = float(
        getattr(args, "soft_tail_hint_step_fraction", DEFAULT_SOFT_TAIL_HINT_STEP_FRACTION)
    )
    if not 0.0 <= soft_tail_hint_step_fraction <= 1.0:
        raise ValueError(
            "soft_tail_hint_step_fraction must be in [0, 1], "
            f"got {soft_tail_hint_step_fraction}."
        )


def build_codec_run_tag(section_span_latents: int, anchor_span_latents: int, num_inference_steps: int) -> str:
    """生成带 codec / 恢复步数的稳定 run 标识。"""
    return (
        f"section{int(section_span_latents)}"
        f"_anchor{int(anchor_span_latents)}"
        f"_steps{int(num_inference_steps)}"
    )


def resolve_train_output_dir(args: argparse.Namespace, codec_config: CodecConfig) -> Path:
    """把 section / anchor / steps 自动拼到训练输出目录名里，便于区分权重。"""
    base_output_dir = Path(args.output_dir)
    run_tag = build_codec_run_tag(
        section_span_latents=codec_config.section_span_latents,
        anchor_span_latents=codec_config.anchor_span_latents,
        num_inference_steps=args.num_inference_steps,
    )
    if base_output_dir.name.endswith(run_tag):
        return base_output_dir
    return base_output_dir.parent / f"{base_output_dir.name}_{run_tag}"


def read_model_path_from_payload(latent_path: Path) -> Optional[str]:
    """尽量从 latent payload 中复用编码阶段记录的模型路径。"""
    try:
        payload = load_payload(latent_path)
    except Exception:
        return None
    model_path = payload.get("model_path")
    if model_path is None:
        return None
    model_path = str(model_path).strip()
    return model_path or None


def resolve_local_base_model_path(
    requested_path: Optional[str],
    *,
    checkpoint_path: Optional[str] = None,
    latent_paths: Optional[Sequence[Path]] = None,
) -> str:
    """解析并校验 recover 主链路使用的本地 Helios base model 目录。"""
    candidate_values: List[str] = []
    seen_values = set()

    def add_candidate(value: Optional[str]) -> None:
        if value is None:
            return
        normalized = str(value).strip()
        if not normalized or normalized in seen_values:
            return
        seen_values.add(normalized)
        candidate_values.append(normalized)

    add_candidate(requested_path)
    add_candidate(os.environ.get("HELIOS_BASE_MODEL_PATH"))
    add_candidate(checkpoint_path)

    if latent_paths is not None:
        payload_model_paths = set()
        for latent_path in latent_paths[:8]:
            payload_model_path = read_model_path_from_payload(latent_path)
            if payload_model_path is not None:
                payload_model_paths.add(payload_model_path)
        if len(payload_model_paths) > 1:
            raise ValueError(
                "Latent payloads contain inconsistent model_path metadata: "
                f"{sorted(payload_model_paths)}. Please pass --base_model_path explicitly."
            )
        if payload_model_paths:
            add_candidate(next(iter(payload_model_paths)))

    if not candidate_values:
        raise FileNotFoundError(
            "Unable to resolve a local Helios base model path. "
            "Please pass --base_model_path <local_dir> or export HELIOS_BASE_MODEL_PATH."
        )

    missing_details: List[str] = []
    for candidate in candidate_values:
        candidate_path = Path(candidate).expanduser()
        if not candidate_path.exists():
            missing_details.append(f"{candidate} (not found)")
            continue
        if not candidate_path.is_dir():
            missing_details.append(f"{candidate_path.resolve()} (not a directory)")
            continue
        missing_subdirs = [
            subdir for subdir in REQUIRED_BASE_MODEL_SUBDIRS if not (candidate_path / subdir).exists()
        ]
        if missing_subdirs:
            missing_details.append(
                f"{candidate_path.resolve()} (missing subdirs: {', '.join(missing_subdirs)})"
            )
            continue
        return str(candidate_path.resolve())

    raise FileNotFoundError(
        "Unable to find a usable local Helios base model directory. Checked: "
        f"{'; '.join(missing_details)}. "
        "Please pass --base_model_path to the directory that contains tokenizer/, text_encoder/, "
        "transformer/, and scheduler/."
    )


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


def build_train_logging_config(
    output_dir: Path,
    args: argparse.Namespace,
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    checkpoint_save_interval_epochs: int,
) -> Dict[str, object]:
    return {
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
        "aux_loss_clip_value": args.aux_loss_clip_value,
        "aux_loss_sigma_min": args.aux_loss_sigma_min,
        "aux_loss_sigma_max": args.aux_loss_sigma_max,
        "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        "soft_tail_hint_strength": args.soft_tail_hint_strength,
        "soft_tail_hint_step_fraction": args.soft_tail_hint_step_fraction,
        "init_checkpoint_dir": None if args.init_checkpoint_dir is None else str(args.init_checkpoint_dir.resolve()),
        "train_mixed_dual_tail_spatial_factor": args.train_mixed_dual_tail_spatial_factor,
        "train_mixed_dual_tail_ratio": args.train_mixed_dual_tail_ratio,
    }


def init_tensorboard_writer(
    output_dir: Path,
    args: argparse.Namespace,
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    checkpoint_save_interval_epochs: int,
):
    """初始化 TensorBoard writer，把训练曲线写到 output_dir/tensorboard。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise ImportError("Recover training requires tensorboard for TensorBoard loss logging.") from exc

    log_dir = output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=str(log_dir))
    config = build_train_logging_config(
        output_dir=output_dir,
        args=args,
        codec_config=codec_config,
        history_sizes=history_sizes,
        latent_window_size=latent_window_size,
        checkpoint_save_interval_epochs=checkpoint_save_interval_epochs,
    )
    writer.add_text("config/json", json.dumps(config, indent=2, ensure_ascii=False), global_step=0)
    return writer


def log_tensorboard_scalars(writer, scalars: Dict[str, object], step: int) -> None:
    if writer is None:
        return
    for key, value in scalars.items():
        if isinstance(value, bool):
            writer.add_scalar(key, int(value), step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(key, value, step)


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
    init_checkpoint_dir = None
    init_checkpoint_config = None
    if args.init_checkpoint_dir is not None:
        init_checkpoint_dir = resolve_checkpoint_dir_for_inference(args.init_checkpoint_dir.resolve())
        init_checkpoint_config = load_recover_config(init_checkpoint_dir)
    args.base_model_path = resolve_local_base_model_path(
        args.base_model_path,
        checkpoint_path=None if init_checkpoint_config is None else init_checkpoint_config.get("base_model_path"),
        latent_paths=latent_paths,
    )
    # 只建立窗口索引，不把所有 latent tensor 常驻 CPU 内存。
    sequence_entries = build_sequence_index(latent_paths)
    dataset = LatentWindowDataset(
        sequences=sequence_entries,
        history_sizes=history_sizes,
        latent_window_size=latent_window_size,
        anchor_span_latents=codec_config.anchor_span_latents,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training windows were built from the provided latent files.")
    sequence_store = LazyPreparedSequenceStore(
        entries=sequence_entries,
        codec_config=codec_config,
        cache_size=args.lazy_sequence_cache_size,
    )

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
        checkpoint_dir=init_checkpoint_dir,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_after_load=False,
    )
    transformer.train()
    learned_tail_codec = None
    learned_codec_trainable_params = 0
    if codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE:
        learned_tail_codec = load_learned_tail_codec(
            checkpoint_dir=init_checkpoint_dir,
            codec_config=codec_config,
            latent_channels=infer_latent_channels(sequence_entries),
            device=device,
        )
        if learned_tail_codec is None:
            learned_tail_codec = build_learned_tail_codec(
                codec_config=asdict(codec_config),
                in_channels=infer_latent_channels(sequence_entries),
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

    output_dir = resolve_train_output_dir(args, codec_config).resolve()
    args.output_dir = output_dir
    checkpoint_save_interval_epochs = max(1, math.ceil(args.epochs / 10)) # 这里记录了多少个epoch保存一下权重
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[train] input_root={input_root} files={len(latent_paths)} windows={len(dataset)} "
            f"device={device} world_size={accelerator.num_processes} "
            f"effective_global_batch_size={args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps} "
            f"lazy_sequence_cache_size={args.lazy_sequence_cache_size} "
            f"tail_codec_type={codec_config.tail_codec_type} "
            f"init_checkpoint_dir={init_checkpoint_dir} "
            f"save_every_epochs={checkpoint_save_interval_epochs} "
            f"checkpoint_every_steps={args.checkpoint_every_steps} "
            f"loss_weighting_scheme={args.loss_weighting_scheme} "
            f"flow/x/noise/delta=({args.flow_loss_weight:.3f}/{args.x_loss_weight:.3f}/{args.noise_loss_weight:.3f}/{args.temporal_delta_loss_weight:.3f}) "
            f"aux_warmup={args.aux_loss_warmup_steps} aux_ramp={args.aux_loss_ramp_steps} "
            f"aux_clip={args.aux_loss_clip_value} aux_sigma=({args.aux_loss_sigma_min}, {args.aux_loss_sigma_max}) "
            f"mixed_dual_tail=({args.train_mixed_dual_tail_spatial_factor}, ratio={args.train_mixed_dual_tail_ratio:.3f}) "
            f"soft_tail_hint=(strength={args.soft_tail_hint_strength:.3f}, "
            f"step_fraction={args.soft_tail_hint_step_fraction:.3f})"
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
        "num_sequences": len(sequence_entries),
        "lazy_sequence_cache_size": args.lazy_sequence_cache_size,
        "loss_weighting_scheme": args.loss_weighting_scheme,
        "logit_mean": args.logit_mean,
        "logit_std": args.logit_std,
        "mode_scale": args.mode_scale,
        "loss_ema_decay": args.loss_ema_decay,
        "num_inference_steps": args.num_inference_steps,
        "flow_loss_weight": args.flow_loss_weight,
        "x_loss_weight": args.x_loss_weight,
        "noise_loss_weight": args.noise_loss_weight,
        "temporal_delta_loss_weight": args.temporal_delta_loss_weight,
        "aux_loss_warmup_steps": args.aux_loss_warmup_steps,
        "aux_loss_ramp_steps": args.aux_loss_ramp_steps,
        "aux_loss_clip_value": args.aux_loss_clip_value,
        "aux_loss_sigma_min": args.aux_loss_sigma_min,
        "aux_loss_sigma_max": args.aux_loss_sigma_max,
        "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        "soft_tail_hint_strength": args.soft_tail_hint_strength,
        "soft_tail_hint_step_fraction": args.soft_tail_hint_step_fraction,
        "tail_codec_type": codec_config.tail_codec_type,
        "init_checkpoint_dir": None if init_checkpoint_dir is None else str(init_checkpoint_dir),
        "train_mixed_dual_tail_spatial_factor": args.train_mixed_dual_tail_spatial_factor,
        "train_mixed_dual_tail_ratio": args.train_mixed_dual_tail_ratio,
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
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "saved_checkpoints": [],
    }

    # 训练日志既保留逐 step 记录，也保留逐 epoch 汇总，方便看短期波动和长期趋势。
    progress = None
    tb_writer = None
    try:
        if accelerator.is_main_process:
            tb_writer = init_tensorboard_writer(
                output_dir=output_dir,
                args=args,
                codec_config=codec_config,
                history_sizes=history_sizes,
                latent_window_size=latent_window_size,
                checkpoint_save_interval_epochs=checkpoint_save_interval_epochs,
            )
            print(f"[train] tensorboard_log_dir={output_dir / 'tensorboard'}")

        if args.max_steps is not None and accelerator.is_main_process:
            progress = tqdm(total=args.max_steps, desc="Training")

        global_step = 0
        stop_training = False
        running_metrics: Dict[str, torch.Tensor] = {}
        running_micro_steps = 0
        loss_ema_value: Optional[float] = None
        last_checkpoint_step: Optional[int] = None
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
                        sequences=sequence_store,
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
                        aux_loss_clip_value=args.aux_loss_clip_value,
                        aux_loss_sigma_min=args.aux_loss_sigma_min,
                        aux_loss_sigma_max=args.aux_loss_sigma_max,
                        soft_tail_hint_strength=args.soft_tail_hint_strength,
                        train_mixed_dual_tail_spatial_factor=args.train_mixed_dual_tail_spatial_factor,
                        train_mixed_dual_tail_ratio=args.train_mixed_dual_tail_ratio,
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
                    log_tensorboard_scalars(
                        tb_writer,
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
                        global_step,
                    )

                should_save_step_checkpoint = (
                    args.checkpoint_every_steps is not None
                    and global_step > 0
                    and global_step % int(args.checkpoint_every_steps) == 0
                    and last_checkpoint_step != global_step
                )
                if should_save_step_checkpoint:
                    step_checkpoint_dir = build_periodic_checkpoint_dir(
                        output_dir=output_dir,
                        epoch=epoch + 1,
                        global_step=global_step,
                    )
                    if accelerator.is_main_process:
                        checkpoint_record = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "checkpoint_dir": str(step_checkpoint_dir),
                            "reason": "step",
                        }
                        train_metrics["saved_checkpoints"].append(checkpoint_record)
                        print(
                            f"[train] saving step checkpoint epoch={epoch + 1} "
                            f"step={global_step} dir={step_checkpoint_dir}"
                        )
                    save_training_artifacts(
                        output_dir=step_checkpoint_dir,
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
                    last_checkpoint_step = global_step
                    if accelerator.is_main_process:
                        log_tensorboard_scalars(
                            tb_writer,
                            {
                                "checkpoint/epoch": epoch + 1,
                                "checkpoint/global_step": global_step,
                                "checkpoint/is_step_checkpoint": 1,
                            },
                            global_step,
                        )
                        if tb_writer is not None:
                            tb_writer.add_text("checkpoint/reason", "step", global_step)
                            tb_writer.flush()

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

            # 周期性 checkpoint 方便观察不同训练阶段的恢复质量，也利于中断后续训。
            should_save_periodic_checkpoint = (
                global_step > 0
                and (epoch + 1) % checkpoint_save_interval_epochs == 0
                and last_checkpoint_step != global_step
            )
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
                log_tensorboard_scalars(
                    tb_writer,
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
                    global_step,
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
                        "reason": "epoch",
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
                last_checkpoint_step = global_step
                if accelerator.is_main_process:
                    log_tensorboard_scalars(
                        tb_writer,
                        {
                            "checkpoint/epoch": epoch + 1,
                            "checkpoint/global_step": global_step,
                            "checkpoint/is_epoch_checkpoint": 1,
                        },
                        global_step,
                    )
                    if tb_writer is not None:
                        tb_writer.add_text("checkpoint/reason", "epoch", global_step)
                        tb_writer.flush()

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
            log_tensorboard_scalars(
                tb_writer,
                {
                    "summary/steps": train_metrics["steps"],
                    "summary/last_loss": train_metrics["losses"][-1],
                    "summary/last_raw_loss": train_metrics["raw_losses"][-1],
                    "summary/last_flow_loss": train_metrics["flow_losses"][-1],
                    "summary/last_x_loss": train_metrics["x_losses"][-1],
                    "summary/last_noise_loss": train_metrics["noise_losses"][-1],
                    "summary/last_temporal_delta_loss": train_metrics["temporal_delta_losses"][-1],
                    "summary/last_loss_ema": train_metrics["losses_ema"][-1],
                    "summary/trainable_params": train_metrics["trainable_params"],
                },
                int(train_metrics["steps"]),
            )
            if tb_writer is not None:
                tb_writer.flush()
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
        if accelerator.is_main_process and tb_writer is not None:
            tb_writer.close()


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
        base_model_path = resolve_local_base_model_path(
            str(base_model_path),
            checkpoint_path=checkpoint_config.get("base_model_path"),
            latent_paths=latent_paths if "latent_paths" in locals() else None,
        )
        args.base_model_path = base_model_path
        weight_dtype_name = resolve_infer_value(
            cli_value=args.weight_dtype,
            default_value=DEFAULT_WEIGHT_DTYPE,
            checkpoint_value=checkpoint_config.get("weight_dtype"),
        )
        args.num_inference_steps = int(
            resolve_infer_value(
                cli_value=args.num_inference_steps,
                default_value=DEFAULT_NUM_INFERENCE_STEPS,
                checkpoint_value=checkpoint_config.get("num_inference_steps"),
            )
        )
        if args.num_inference_steps <= 0:
            raise ValueError(f"--num_inference_steps must be > 0, got {args.num_inference_steps}.")
        mode_inference_steps = resolve_mode_inference_steps(args, checkpoint_config)
        codec_config = CodecConfig(**checkpoint_config["codec_config"])
        codec_config.single_head_codec_type = str(
            resolve_infer_value(
                cli_value=args.single_head_codec_type,
                default_value=DEFAULT_SINGLE_HEAD_CODEC_TYPE,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "single_head_codec_type",
                    DEFAULT_SINGLE_HEAD_CODEC_TYPE,
                ),
            )
        )
        if codec_config.single_head_codec_type not in {
            SINGLE_HEAD_ANCHOR_CODEC_TYPE,
            SINGLE_HEAD_SOFT_POOL_CODEC_TYPE,
        }:
            raise ValueError(
                "--single_head_codec_type must be one of "
                f"{SINGLE_HEAD_ANCHOR_CODEC_TYPE}, {SINGLE_HEAD_SOFT_POOL_CODEC_TYPE}; "
                f"got {codec_config.single_head_codec_type}."
            )
        codec_config.single_head_soft_pool_spatial_factor = int(
            resolve_infer_value(
                cli_value=args.single_head_soft_pool_spatial_factor,
                default_value=DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "single_head_soft_pool_spatial_factor",
                    DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
                ),
            )
        )
        if codec_config.single_head_soft_pool_spatial_factor < 1:
            raise ValueError(
                "--single_head_soft_pool_spatial_factor must be >= 1, "
                f"got {codec_config.single_head_soft_pool_spatial_factor}."
            )
        codec_config.dual_head_codec_type = str(
            resolve_infer_value(
                cli_value=args.dual_head_codec_type,
                default_value=DEFAULT_DUAL_HEAD_CODEC_TYPE,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "dual_head_codec_type",
                    DEFAULT_DUAL_HEAD_CODEC_TYPE,
                ),
            )
        )
        if codec_config.dual_head_codec_type not in {
            DUAL_HEAD_ANCHOR_CODEC_TYPE,
            DUAL_HEAD_P_DELTA_CODEC_TYPE,
            DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
        }:
            raise ValueError(
                "--dual_head_codec_type must be one of "
                f"{DUAL_HEAD_ANCHOR_CODEC_TYPE}, {DUAL_HEAD_P_DELTA_CODEC_TYPE}, "
                f"{DUAL_HEAD_SOFT_POOL_CODEC_TYPE}; "
                f"got {codec_config.dual_head_codec_type}."
            )
        resolved_dual_head_anchor_spatial_factor = resolve_infer_value(
            cli_value=args.dual_head_anchor_spatial_factor,
            default_value=DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR,
            checkpoint_value=checkpoint_config.get("codec_config", {}).get("dual_head_anchor_spatial_factor"),
        )
        codec_config.dual_head_anchor_spatial_factor = (
            None
            if resolved_dual_head_anchor_spatial_factor is None
            else int(resolved_dual_head_anchor_spatial_factor)
        )
        if (
            codec_config.dual_head_anchor_spatial_factor is not None
            and codec_config.dual_head_anchor_spatial_factor < 1
        ):
            raise ValueError(
                "--dual_head_anchor_spatial_factor must be >= 1, "
                f"got {codec_config.dual_head_anchor_spatial_factor}."
            )
        codec_config.dual_head_p_delta_spatial_factor = int(
            resolve_infer_value(
                cli_value=args.dual_head_p_delta_spatial_factor,
                default_value=DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "dual_head_p_delta_spatial_factor",
                    DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
                ),
            )
        )
        if codec_config.dual_head_p_delta_spatial_factor < 1:
            raise ValueError(
                "--dual_head_p_delta_spatial_factor must be >= 1, "
                f"got {codec_config.dual_head_p_delta_spatial_factor}."
            )
        codec_config.dual_head_soft_pool_spatial_factor = int(
            resolve_infer_value(
                cli_value=args.dual_head_soft_pool_spatial_factor,
                default_value=DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "dual_head_soft_pool_spatial_factor",
                    DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
                ),
            )
        )
        if codec_config.dual_head_soft_pool_spatial_factor < 1:
            raise ValueError(
                "--dual_head_soft_pool_spatial_factor must be >= 1, "
                f"got {codec_config.dual_head_soft_pool_spatial_factor}."
            )
        codec_config.adaptive_dual_head_full_ratio = float(
            resolve_infer_value(
                cli_value=args.adaptive_dual_head_full_ratio,
                default_value=DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "adaptive_dual_head_full_ratio",
                    DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
                ),
            )
        )
        if not 0.0 <= codec_config.adaptive_dual_head_full_ratio <= 1.0:
            raise ValueError(
                "--adaptive_dual_head_full_ratio must be in [0, 1], "
                f"got {codec_config.adaptive_dual_head_full_ratio}."
            )
        resolved_adaptive_dual_head_soft_pool_hard_factor = resolve_infer_value(
            cli_value=args.adaptive_dual_head_soft_pool_hard_factor,
            default_value=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
            checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                "adaptive_dual_head_soft_pool_hard_factor",
                DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
            ),
        )
        codec_config.adaptive_dual_head_soft_pool_hard_factor = (
            None
            if resolved_adaptive_dual_head_soft_pool_hard_factor is None
            else int(resolved_adaptive_dual_head_soft_pool_hard_factor)
        )
        if (
            codec_config.adaptive_dual_head_soft_pool_hard_factor is not None
            and codec_config.adaptive_dual_head_soft_pool_hard_factor < 1
        ):
            raise ValueError(
                "--adaptive_dual_head_soft_pool_hard_factor must be >= 1 when set, "
                f"got {codec_config.adaptive_dual_head_soft_pool_hard_factor}."
            )
        codec_config.adaptive_dual_head_soft_pool_hard_ratio = float(
            resolve_infer_value(
                cli_value=args.adaptive_dual_head_soft_pool_hard_ratio,
                default_value=DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "adaptive_dual_head_soft_pool_hard_ratio",
                    DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
                ),
            )
        )
        if not 0.0 <= codec_config.adaptive_dual_head_soft_pool_hard_ratio <= 1.0:
            raise ValueError(
                "--adaptive_dual_head_soft_pool_hard_ratio must be in [0, 1], "
                f"got {codec_config.adaptive_dual_head_soft_pool_hard_ratio}."
            )
        codec_config.global_keyframe_codec_type = str(
            resolve_infer_value(
                cli_value=args.global_keyframe_codec_type,
                default_value=DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "global_keyframe_codec_type",
                    DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE,
                ),
            )
        )
        codec_config.global_keyframe_quant_dtype = str(
            resolve_infer_value(
                cli_value=args.global_keyframe_quant_dtype,
                default_value=DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "global_keyframe_quant_dtype",
                    DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE,
                ),
            )
        )
        codec_config.global_keyframe_spatial_factor = int(
            resolve_infer_value(
                cli_value=args.global_keyframe_spatial_factor,
                default_value=DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "global_keyframe_spatial_factor",
                    DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR,
                ),
            )
        )
        resolved_dual_tail_anchor_spatial_factor = resolve_infer_value(
            cli_value=args.dual_tail_anchor_spatial_factor,
            default_value=DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR,
            checkpoint_value=checkpoint_config.get("codec_config", {}).get("dual_tail_anchor_spatial_factor"),
        )
        codec_config.dual_tail_anchor_spatial_factor = (
            None
            if resolved_dual_tail_anchor_spatial_factor is None
            else int(resolved_dual_tail_anchor_spatial_factor)
        )
        if (
            codec_config.dual_tail_anchor_spatial_factor is not None
            and codec_config.dual_tail_anchor_spatial_factor < 1
        ):
            raise ValueError(
                "--dual_tail_anchor_spatial_factor must be >= 1, "
                f"got {codec_config.dual_tail_anchor_spatial_factor}."
            )
        codec_config.dual_tail_codec_type = str(
            resolve_infer_value(
                cli_value=args.dual_tail_codec_type,
                default_value=DEFAULT_DUAL_TAIL_CODEC_TYPE,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get("dual_tail_codec_type"),
            )
        )
        codec_config.dual_tail_p_delta_spatial_factor = int(
            resolve_infer_value(
                cli_value=args.dual_tail_p_delta_spatial_factor,
                default_value=DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "dual_tail_p_delta_spatial_factor",
                    DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
                ),
            )
        )
        resolved_target_low_bpp = resolve_infer_value(
            cli_value=args.target_low_bpp,
            default_value=DEFAULT_TARGET_LOW_BPP,
            checkpoint_value=checkpoint_config.get("codec_config", {}).get("target_low_bpp"),
        )
        codec_config.target_low_bpp = (
            None if resolved_target_low_bpp is None else float(resolved_target_low_bpp)
        )
        if codec_config.target_low_bpp is not None and codec_config.target_low_bpp <= 0.0:
            raise ValueError(f"--target_low_bpp must be > 0, got {codec_config.target_low_bpp}.")
        codec_config.max_predict_only_gap_sections = int(
            resolve_infer_value(
                cli_value=args.max_predict_only_gap_sections,
                default_value=DEFAULT_MAX_PREDICT_ONLY_GAP,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "max_predict_only_gap_sections",
                    DEFAULT_MAX_PREDICT_ONLY_GAP,
                ),
            )
        )
        codec_config.single_refresh_gain_threshold = float(
            resolve_infer_value(
                cli_value=args.single_refresh_gain_threshold,
                default_value=DEFAULT_SINGLE_REFRESH_GAIN,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "single_refresh_gain_threshold",
                    DEFAULT_SINGLE_REFRESH_GAIN,
                ),
            )
        )
        codec_config.dual_refresh_gain_threshold = float(
            resolve_infer_value(
                cli_value=args.dual_refresh_gain_threshold,
                default_value=DEFAULT_DUAL_REFRESH_GAIN,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "dual_refresh_gain_threshold",
                    DEFAULT_DUAL_REFRESH_GAIN,
                ),
            )
        )
        codec_config.boundary_jump_threshold = float(
            resolve_infer_value(
                cli_value=args.boundary_jump_threshold,
                default_value=DEFAULT_BOUNDARY_JUMP,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "boundary_jump_threshold",
                    DEFAULT_BOUNDARY_JUMP,
                ),
            )
        )
        codec_config.cut_detection_threshold = float(
            resolve_infer_value(
                cli_value=args.cut_detection_threshold,
                default_value=DEFAULT_CUT_DETECTION,
                checkpoint_value=checkpoint_config.get("codec_config", {}).get(
                    "cut_detection_threshold",
                    DEFAULT_CUT_DETECTION,
                ),
            )
        )
        validate_codec_config(asdict(codec_config))
        history_sizes = normalize_history_sizes(checkpoint_config["history_sizes"])
        latent_window_size = int(checkpoint_config.get("section_span_latents", checkpoint_config["latent_window_size"]))
        anchor_span_latents = int(checkpoint_config.get("anchor_span_latents", codec_config.anchor_span_latents))
        fix_anchor_during_denoise = bool(
            checkpoint_config.get("fix_anchor_during_denoise", DEFAULT_FIX_ANCHOR_DURING_DENOISE)
        )
        soft_tail_hint_strength = float(
            resolve_infer_value(
                cli_value=args.soft_tail_hint_strength,
                default_value=DEFAULT_SOFT_TAIL_HINT_STRENGTH,
                checkpoint_value=checkpoint_config.get("soft_tail_hint_strength"),
            )
        )
        if soft_tail_hint_strength < 0.0:
            raise ValueError(f"--soft_tail_hint_strength must be >= 0, got {soft_tail_hint_strength}.")
        soft_tail_hint_step_fraction = float(
            resolve_infer_value(
                cli_value=args.soft_tail_hint_step_fraction,
                default_value=DEFAULT_SOFT_TAIL_HINT_STEP_FRACTION,
                checkpoint_value=checkpoint_config.get("soft_tail_hint_step_fraction"),
            )
        )
        if not 0.0 <= soft_tail_hint_step_fraction <= 1.0:
            raise ValueError(
                "--soft_tail_hint_step_fraction must be in [0, 1], "
                f"got {soft_tail_hint_step_fraction}."
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
            print(f"[infer] mode_inference_steps={mode_inference_steps}")
            print(f"[infer] dual_tail_anchor_spatial_factor={codec_config.dual_tail_anchor_spatial_factor}")
            print(f"[infer] target_low_bpp={codec_config.target_low_bpp}")
            print(
                f"[infer] soft_tail_hint=(strength={soft_tail_hint_strength:.3f}, "
                f"step_fraction={soft_tail_hint_step_fraction:.3f})"
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
        per_video_latency_seconds: List[float] = []

        for sample_idx, latent_path in enumerate(assigned_latent_paths, start=1):
            sample_start_time = start_synced_timer(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            print(
                f"[infer][rank={distributed_context.rank}] ({sample_idx}/{len(assigned_latent_paths)}) "
                f"input={latent_path}"
            )

            prepare_timing: Dict[str, float] = {}
            stage_prepare_start = start_synced_timer(device)
            sequence = prepare_sequence(
                latent_path,
                codec_config,
                learned_tail_codec=learned_tail_codec,
                materialize_low_latents=True,
                use_ste_quant=False,
                move_to_cpu=True,
                timing_device=device,
                timing=prepare_timing,
            )
            stage_prepare_seconds = stop_synced_timer(stage_prepare_start, device)

            reconstruct_timing: Dict[str, object] = {}
            stage_reconstruct_start = start_synced_timer(device)
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
                mode_inference_steps=mode_inference_steps,
                seed=args.seed,
                fix_anchor_during_denoise=fix_anchor_during_denoise,
                soft_tail_hint_strength=soft_tail_hint_strength,
                soft_tail_hint_step_fraction=soft_tail_hint_step_fraction,
                distributed_context=distributed_context,
                timing=reconstruct_timing,
            )
            stage_reconstruct_seconds = stop_synced_timer(stage_reconstruct_start, device)

            postprocess_timing: Dict[str, float] = {}
            stage_postprocess_start = start_synced_timer(device)
            resolve_paths_start_time = start_synced_timer(device)
            low_latent_path, recover_latent_path, metrics_path = resolve_output_paths(
                input_path=latent_path,
                input_root=input_root,
                low_dir=low_dir,
                recover_dir=recover_dir,
                metrics_dir=metrics_dir,
            )
            postprocess_timing["resolve_output_paths_seconds"] = stop_synced_timer(resolve_paths_start_time, device)
            save_low_start_time = start_synced_timer(device)
            low_payload = build_low_latent_payload(sequence)
            torch.save(low_payload, low_latent_path)
            postprocess_timing["build_and_save_low_latents_seconds"] = stop_synced_timer(save_low_start_time, device)
            save_recover_start_time = start_synced_timer(device)
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
            postprocess_timing["build_and_save_recover_latents_seconds"] = stop_synced_timer(
                save_recover_start_time,
                device,
            )

            metrics_compute_start_time = start_synced_timer(device)
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
            postprocess_timing["compute_metrics_seconds"] = stop_synced_timer(metrics_compute_start_time, device)
            gpu_peak_memory = None
            if device.type == "cuda":
                gpu_peak_memory = {
                    "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            source_duration_seconds = (
                float(sequence.metadata.source_num_frames) / float(sequence.metadata.source_fps)
                if sequence.metadata.source_fps > 0
                else 0.0
            )
            first_section_ready_seconds = reconstruct_timing.get("first_section_ready_seconds")
            first_non_keyframe_ready_seconds = (
                None
                if first_section_ready_seconds is None
                else stage_prepare_seconds + float(first_section_ready_seconds)
            )
            streaming_frame_group_ready_events = []
            for event in reconstruct_timing.get("streaming_latent_ready_events", []):
                reconstruct_ready_seconds = float(event["ready_seconds"])
                enriched_event = dict(event)
                enriched_event["reconstruct_ready_seconds"] = reconstruct_ready_seconds
                enriched_event["ready_seconds"] = stage_prepare_seconds + reconstruct_ready_seconds
                streaming_frame_group_ready_events.append(enriched_event)
            metrics_payload = {
                "input_path": str(sequence.path),
                "low_latent_path": str(low_latent_path),
                "recover_latent_path": str(recover_latent_path),
                "raw_file_bytes": sequence.metadata.raw_file_bytes,
                "raw_bpp": sequence.metadata.raw_bpp,
                "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
                "global_keyframe_bytes": int(sequence.low_codec_payload["global_keyframe_bytes"]),
                "single_head_bytes": int(sequence.low_codec_payload["single_head_bytes"]),
                "dual_head_bytes": int(sequence.low_codec_payload["dual_head_bytes"]),
                "dual_tail_presidual_bytes": int(sequence.low_codec_payload["dual_tail_presidual_bytes"]),
                "metadata_bytes": int(sequence.low_codec_payload["metadata_bytes"]),
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
                "streaming_frame_group_ready_events": streaming_frame_group_ready_events,
                "streaming_frame_group_ready_seconds": [
                    float(event["ready_seconds"]) for event in streaming_frame_group_ready_events
                ],
                **temporal_backtrack_metrics,
                "codec_config": asdict(codec_config),
                "checkpoint_dir": str(checkpoint_dir),
                "num_inference_steps": args.num_inference_steps,
                "num_inference_steps_by_mode": {key: int(value) for key, value in mode_inference_steps.items()},
                "effective_inference_steps": count_effective_inference_steps(
                    sequence.low_codec_payload,
                    mode_inference_steps,
                ),
                "section_mode_counts": summarize_section_modes(sequence.low_codec_payload),
                "mode_selection_strategy": sequence.low_codec_payload.get("mode_selection_strategy"),
                "target_low_bpp": sequence.low_codec_payload.get("target_low_bpp"),
                "target_low_bytes": sequence.low_codec_payload.get("target_low_bytes"),
                "estimated_selected_bytes": sequence.low_codec_payload.get("estimated_selected_bytes"),
                "estimated_fixed_bytes": sequence.low_codec_payload.get("estimated_fixed_bytes"),
                "latency_seconds": {
                    "total": None,
                    "prepare_low_latents": stage_prepare_seconds,
                    "reconstruct": stage_reconstruct_seconds,
                    "postprocess_io_and_metrics": None,
                    "first_keyframe_ready": stage_prepare_seconds,
                    "first_non_keyframe_ready": first_non_keyframe_ready_seconds,
                },
                "prepare_timing_seconds": prepare_timing,
                "reconstruct_timing": reconstruct_timing,
                "postprocess_timing_seconds": postprocess_timing,
                "gpu_peak_memory": gpu_peak_memory,
                "source_duration_seconds": source_duration_seconds,
                "realtime_factor": None,
            }
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            write_metrics_start_time = start_synced_timer(device)
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)
            postprocess_timing["write_metrics_json_seconds"] = stop_synced_timer(write_metrics_start_time, device)
            stage_postprocess_seconds = stop_synced_timer(stage_postprocess_start, device)
            sample_total_seconds = stop_synced_timer(sample_start_time, device)
            postprocess_timing["total_seconds"] = stage_postprocess_seconds
            metrics_payload["latency_seconds"]["total"] = sample_total_seconds
            metrics_payload["latency_seconds"]["postprocess_io_and_metrics"] = stage_postprocess_seconds
            metrics_payload["postprocess_timing_seconds"] = postprocess_timing
            realtime_factor = (
                sample_total_seconds / source_duration_seconds if source_duration_seconds > 0.0 else None
            )
            metrics_payload["realtime_factor"] = realtime_factor
            with metrics_path.open("w", encoding="utf-8") as handle:
                json.dump(metrics_payload, handle, indent=2)

            per_video_latency_seconds.append(sample_total_seconds)
            realtime_factor_text = "n/a" if realtime_factor is None else f"{realtime_factor:.3f}x"
            gpu_peak_mem_text = "n/a"
            if gpu_peak_memory is not None:
                alloc_gib = gpu_peak_memory["max_memory_allocated_bytes"] / (1024 ** 3)
                reserve_gib = gpu_peak_memory["max_memory_reserved_bytes"] / (1024 ** 3)
                gpu_peak_mem_text = f"alloc={alloc_gib:.3f}GiB reserved={reserve_gib:.3f}GiB"
            first_non_keyframe_text = (
                "n/a" if first_non_keyframe_ready_seconds is None else f"{first_non_keyframe_ready_seconds:.3f}s"
            )

            print(
                f"[infer][rank={distributed_context.rank}] saved low={low_latent_path} recover={recover_latent_path} "
                f"low_bpp={sequence.low_codec_payload['low_bpp']:.6f} "
                f"effective_steps={metrics_payload['effective_inference_steps']} "
                f"direct_low_l1={direct_low_metrics['l1']:.6f} restored_l1={restored_metrics['l1']:.6f} "
                f"latency={sample_total_seconds:.3f}s (prepare={stage_prepare_seconds:.3f}s "
                f"reconstruct={stage_reconstruct_seconds:.3f}s post={stage_postprocess_seconds:.3f}s "
                f"first_non_keyframe={first_non_keyframe_text}) "
                f"rtf={realtime_factor_text} gpu_peak={gpu_peak_mem_text}"
            )

            if device.type == "cuda":
                # 逐样本清 cache，减轻长序列推理时的显存峰值压力。
                torch.cuda.empty_cache()

        if per_video_latency_seconds:
            avg_latency_seconds = sum(per_video_latency_seconds) / len(per_video_latency_seconds)
            print(
                f"[infer][rank={distributed_context.rank}] latency summary: "
                f"videos={len(per_video_latency_seconds)} avg={avg_latency_seconds:.3f}s "
                f"min={min(per_video_latency_seconds):.3f}s max={max(per_video_latency_seconds):.3f}s"
            )
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
        global_keyframe_codec_type=args.global_keyframe_codec_type,
        global_keyframe_quant_dtype=args.global_keyframe_quant_dtype,
        global_keyframe_spatial_factor=args.global_keyframe_spatial_factor,
        section_span_latents=section_span_latents,
        anchor_span_latents=anchor_span_latents,
        tail_span_latents=tail_span_latents,
        anchor_quant_dtype=args.anchor_quant_dtype,
        anchor_spatial_factor=args.anchor_spatial_factor,
        single_head_codec_type=getattr(args, "single_head_codec_type", DEFAULT_SINGLE_HEAD_CODEC_TYPE),
        single_head_soft_pool_spatial_factor=getattr(
            args,
            "single_head_soft_pool_spatial_factor",
            DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        ),
        dual_head_codec_type=getattr(args, "dual_head_codec_type", DEFAULT_DUAL_HEAD_CODEC_TYPE),
        dual_head_anchor_spatial_factor=getattr(args, "dual_head_anchor_spatial_factor", DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR),
        dual_head_p_delta_spatial_factor=getattr(
            args,
            "dual_head_p_delta_spatial_factor",
            DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR,
        ),
        dual_head_soft_pool_spatial_factor=getattr(
            args,
            "dual_head_soft_pool_spatial_factor",
            DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        ),
        adaptive_dual_head_full_ratio=getattr(
            args,
            "adaptive_dual_head_full_ratio",
            DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO,
        ),
        adaptive_dual_head_soft_pool_hard_factor=getattr(
            args,
            "adaptive_dual_head_soft_pool_hard_factor",
            DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR,
        ),
        adaptive_dual_head_soft_pool_hard_ratio=getattr(
            args,
            "adaptive_dual_head_soft_pool_hard_ratio",
            DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
        ),
        dual_tail_anchor_spatial_factor=getattr(args, "dual_tail_anchor_spatial_factor", DEFAULT_DUAL_TAIL_ANCHOR_SPATIAL_FACTOR),
        dual_tail_codec_type=getattr(args, "dual_tail_codec_type", DEFAULT_DUAL_TAIL_CODEC_TYPE),
        dual_tail_p_delta_spatial_factor=getattr(
            args,
            "dual_tail_p_delta_spatial_factor",
            DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR,
        ),
        tail_codec_type=args.tail_codec_type,
        max_predict_only_gap_sections=args.max_predict_only_gap_sections,
        single_refresh_gain_threshold=args.single_refresh_gain_threshold,
        dual_refresh_gain_threshold=args.dual_refresh_gain_threshold,
        boundary_jump_threshold=args.boundary_jump_threshold,
        cut_detection_threshold=args.cut_detection_threshold,
        target_low_bpp=getattr(args, "target_low_bpp", DEFAULT_TARGET_LOW_BPP),
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


def build_sequence_index_entry(path: Path) -> LatentSequenceIndexEntry:
    """读取单个 latent 的轻量索引信息，然后释放 payload tensor。"""
    payload = load_payload(path)
    validate_payload(payload, path)
    latent_chunks = payload["latent_chunks"]
    first_chunk = latent_chunks[0]
    latent_channels = int(first_chunk.shape[0])
    chunk_lengths = [int(chunk.shape[1]) for chunk in latent_chunks]
    chunk_frame_ranges = build_chunk_frame_ranges(
        [int((chunk.shape[1] - 1) * 4 + 1) for chunk in latent_chunks]
    )
    source_num_frames = infer_source_num_frames(latent_chunks, payload)
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
    return LatentSequenceIndexEntry(
        path=path.resolve(),
        metadata=metadata,
        latent_channels=latent_channels,
        total_latent_frames=sum(chunk_lengths),
    )


def build_sequence_index(latent_paths: Sequence[Path]) -> List[LatentSequenceIndexEntry]:
    """构建训练窗口需要的轻量索引，不保留任何 latent tensor。"""
    return [build_sequence_index_entry(path) for path in latent_paths]


def prepare_sequence(
    path: Path,
    codec_config: CodecConfig,
    learned_tail_codec: Optional[torch.nn.Module] = None,
    materialize_low_latents: bool = True,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
    timing_device: Optional[torch.device] = None,
    timing: Optional[Dict[str, float]] = None,
) -> PreparedSequence:
    """把单个 latent 文件预处理成 recover 主链路需要的统一结构。

    这里会同时算出：
    - clean_full_latents: 原始高保真表示
    - low_codec_payload: 低码率编码结果
    - low_full_latents: 低码率解码后的粗恢复结果

    这样训练与推理都能围绕同一份中间表示展开。
    """
    prepare_start_time = start_synced_timer(timing_device) if timing is not None else None
    load_start_time = start_synced_timer(timing_device) if timing is not None else None
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
    load_seconds = (
        stop_synced_timer(load_start_time, timing_device)
        if load_start_time is not None
        else None
    )
    if not materialize_low_latents:
        if timing is not None:
            timing.clear()
            timing["total_seconds"] = stop_synced_timer(prepare_start_time, timing_device)
            timing["load_metadata_and_flatten_seconds"] = 0.0 if load_seconds is None else float(load_seconds)
            timing["materialize_low_latents_seconds"] = 0.0
        return sequence

    materialize_start_time = start_synced_timer(timing_device) if timing is not None else None
    low_latent_timing: Dict[str, float] = {}
    low_codec_payload, low_full_latents = build_sequence_low_latents(
        sequence=sequence,
        codec_config=codec_config,
        learned_tail_codec=learned_tail_codec,
        device=None,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
        timing_device=timing_device,
        timing=low_latent_timing if timing is not None else None,
    )
    sequence.low_codec_payload = low_codec_payload
    sequence.low_full_latents = low_full_latents
    if timing is not None:
        timing.clear()
        timing["total_seconds"] = stop_synced_timer(prepare_start_time, timing_device)
        timing["load_metadata_and_flatten_seconds"] = 0.0 if load_seconds is None else float(load_seconds)
        timing["materialize_low_latents_seconds"] = (
            0.0
            if materialize_start_time is None
            else stop_synced_timer(materialize_start_time, timing_device)
        )
        timing.update(low_latent_timing)
    return sequence


def get_sequence_latent_channels(sequence: PreparedSequence | LatentSequenceIndexEntry) -> int:
    if isinstance(sequence, LatentSequenceIndexEntry):
        return int(sequence.latent_channels)
    return int(sequence.clean_full_latents.shape[0])


def infer_latent_channels(sequences: Sequence[PreparedSequence | LatentSequenceIndexEntry]) -> int:
    if not sequences:
        raise ValueError("Expected at least one sequence to infer latent channels.")
    latent_channels = get_sequence_latent_channels(sequences[0])
    for sequence in sequences[1:]:
        current_channels = get_sequence_latent_channels(sequence)
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
    train_mixed_dual_tail_spatial_factor: Optional[int] = None,
    train_mixed_dual_tail_ratio: float = 0.0,
    timing_device: Optional[torch.device] = None,
    timing: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, object], torch.Tensor]:
    build_start_time = start_synced_timer(timing_device) if timing is not None else None
    clean_full_latents = sequence.clean_full_latents
    move_input_start_time = start_synced_timer(timing_device) if timing is not None else None
    if device is None and learned_tail_codec is not None:
        try:
            device = next(learned_tail_codec.parameters()).device
        except StopIteration:
            device = None
    if device is not None:
        clean_full_latents = clean_full_latents.to(device=device, dtype=torch.float32)
    else:
        clean_full_latents = clean_full_latents.float()
    move_input_seconds = (
        stop_synced_timer(move_input_start_time, timing_device)
        if move_input_start_time is not None
        else None
    )

    # 压缩到低码率模式，模拟真实传输时候的码率
    encode_start_time = start_synced_timer(timing_device) if timing is not None else None
    low_codec_payload = encode_low_latents(
        clean_full_latents=clean_full_latents,
        codec_config=codec_config,
        chunk_lengths=sequence.metadata.chunk_lengths,
        total_pixels=sequence.metadata.total_pixels,
        learned_tail_codec=learned_tail_codec,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
        train_mixed_dual_tail_spatial_factor=train_mixed_dual_tail_spatial_factor,
        train_mixed_dual_tail_ratio=train_mixed_dual_tail_ratio,
    )
    encode_seconds = (
        stop_synced_timer(encode_start_time, timing_device)
        if encode_start_time is not None
        else None
    )
    low_codec_payload["low_codec_bytes"] = estimate_low_codec_bytes(low_codec_payload)
    low_codec_payload.update(estimate_low_codec_byte_breakdown(low_codec_payload))
    low_codec_payload["low_bpp"] = compute_bpp_from_total_pixels(
        int(low_codec_payload["low_codec_bytes"]),
        sequence.metadata.total_pixels,
    )
    # 讲latents恢复到原尺寸
    decode_start_time = start_synced_timer(timing_device) if timing is not None else None
    low_full_latents = decode_low_latents(
        codec_payload=low_codec_payload,
        learned_tail_codec=learned_tail_codec,
    ).float().contiguous()
    decode_seconds = (
        stop_synced_timer(decode_start_time, timing_device)
        if decode_start_time is not None
        else None
    )
    move_output_start_time = start_synced_timer(timing_device) if timing is not None else None
    if move_to_cpu:
        low_full_latents = low_full_latents.cpu().contiguous()
    move_output_seconds = (
        stop_synced_timer(move_output_start_time, timing_device)
        if move_output_start_time is not None
        else None
    )
    if timing is not None:
        timing.clear()
        timing["total_seconds"] = stop_synced_timer(build_start_time, timing_device)
        timing["move_clean_latents_to_device_seconds"] = (
            0.0 if move_input_seconds is None else float(move_input_seconds)
        )
        timing["encode_low_latents_seconds"] = 0.0 if encode_seconds is None else float(encode_seconds)
        timing["decode_low_latents_seconds"] = 0.0 if decode_seconds is None else float(decode_seconds)
        timing["move_low_latents_to_cpu_seconds"] = 0.0 if move_output_seconds is None else float(move_output_seconds)
    return low_codec_payload, low_full_latents


def encode_low_latents(
    clean_full_latents: torch.Tensor,
    codec_config: CodecConfig,
    chunk_lengths: Sequence[int],
    total_pixels: Optional[int] = None,
    learned_tail_codec: Optional[torch.nn.Module] = None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
    train_mixed_dual_tail_spatial_factor: Optional[int] = None,
    train_mixed_dual_tail_ratio: float = 0.0,
) -> Dict[str, object]:
    """把 clean latents 编码成 `x0 + sparse refresh events` 低码率载荷。"""
    codec_payload = encode_anchor_plus_tail_latents(
        clean_full_latents=clean_full_latents,
        codec_config=asdict(codec_config),
        chunk_lengths=chunk_lengths,
        total_pixels=total_pixels,
        learned_tail_codec=learned_tail_codec,
        use_ste_quant=use_ste_quant,
        move_to_cpu=move_to_cpu,
        train_mixed_dual_tail_spatial_factor=train_mixed_dual_tail_spatial_factor,
        train_mixed_dual_tail_ratio=train_mixed_dual_tail_ratio,
    )
    codec_payload["codec_config"] = asdict(codec_config)
    codec_payload["format_version"] = DEFAULT_LOW_LATENT_FORMAT_VERSION
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


def estimate_low_codec_byte_breakdown(codec_payload: Dict[str, object]) -> Dict[str, int]:
    if "section_payloads" in codec_payload:
        return estimate_anchor_plus_tail_codec_byte_breakdown(codec_payload)
    if "section_anchor_payloads" in codec_payload and "section_tail_payloads" in codec_payload:
        return estimate_anchor_plus_tail_codec_byte_breakdown(codec_payload)

    breakdown = {key: 0 for key in LOW_CODEC_BYTE_BREAKDOWN_KEYS}
    keyframe = codec_payload.get("keyframe")
    if isinstance(keyframe, torch.Tensor):
        breakdown["global_keyframe_bytes"] += int(keyframe.numel() * keyframe.element_size())

    for key in ("quantized_remainder", "scales"):
        value = codec_payload.get(key)
        if isinstance(value, torch.Tensor):
            breakdown["single_head_bytes"] += int(value.numel() * value.element_size())

    metadata_values: List[int] = [len(codec_payload["chunk_lengths"])]
    metadata_values.extend(int(value) for value in codec_payload["chunk_lengths"])
    metadata_values.extend(int(value) for value in codec_payload["reduced_shape"])
    metadata_values.extend(int(value) for value in codec_payload["original_remainder_shape"])
    breakdown["metadata_bytes"] += len(metadata_values) * 4
    return {key: int(value) for key, value in breakdown.items()}


def estimate_low_codec_bytes(codec_payload: Dict[str, object]) -> int:
    """估算低码率载荷的字节数。

    这里统计的是 tensor 数据本身加最小必要 shape 元数据，
    用来近似衡量这条低码率链路的传输成本。
    """
    if "section_payloads" in codec_payload:
        return estimate_anchor_plus_tail_codec_bytes(codec_payload)
    if "section_anchor_payloads" in codec_payload and "section_tail_payloads" in codec_payload:
        return estimate_anchor_plus_tail_codec_bytes(codec_payload)
    breakdown = estimate_low_codec_byte_breakdown(codec_payload)
    return int(sum(int(value) for value in breakdown.values()))


def synchronize_device_for_timing(device: Optional[torch.device]) -> None:
    """在需要时同步设备，避免 CUDA 异步导致的时延统计失真。"""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def start_synced_timer(device: Optional[torch.device]) -> float:
    """开启一个同步后的 wall-clock 计时器。"""
    synchronize_device_for_timing(device)
    return time.perf_counter()


def stop_synced_timer(start_time: float, device: Optional[torch.device]) -> float:
    """结束一个同步后的 wall-clock 计时器。"""
    synchronize_device_for_timing(device)
    return time.perf_counter() - start_time


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
    source_full_latents: torch.Tensor,
    section_start: int,
    history_window_size: int,
) -> torch.Tensor:
    """从当前可用的 full latents 中截取历史窗口。

    注意这里故意只取 `low_full_latents[:, 1:]`：
    - 首帧已单独作为 `x0_latents` 输入
    - 历史窗口只负责提供后续低码率上下文
    """
    source_remainder = source_full_latents[:, 1:]
    remainder_start = max(0, (section_start - 1) - history_window_size)
    remainder_end = max(0, section_start - 1)
    history = source_remainder[:, remainder_start:remainder_end]
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


def build_section_payload_lookup(codec_payload: Dict[str, object]) -> Dict[int, Dict[str, object]]:
    if "section_payloads" not in codec_payload or not codec_payload.get("section_payloads"):
        return {}
    lookup: Dict[int, Dict[str, object]] = {}
    for (section_start, _section_end), section_payload in zip(
        codec_payload.get("section_ranges", []),
        codec_payload.get("section_payloads", []),
    ):
        lookup[int(section_start)] = section_payload
    return lookup


def resolve_section_mode(section_payload: Optional[Dict[str, object]]) -> str:
    """把 payload 中记录的 section mode 规范化为已知枚举。"""
    if section_payload is None:
        return SINGLE_REFRESH_SECTION_MODE
    mode_name = str(section_payload.get("mode", SINGLE_REFRESH_SECTION_MODE)).strip()
    if mode_name not in VALID_SECTION_MODES:
        return SINGLE_REFRESH_SECTION_MODE
    return mode_name


def summarize_section_modes(codec_payload: Dict[str, object]) -> Dict[str, int]:
    """统计当前视频各类 section mode 的数量。"""
    counts = {
        PREDICT_ONLY_SECTION_MODE: 0,
        SINGLE_REFRESH_SECTION_MODE: 0,
        DUAL_REFRESH_SECTION_MODE: 0,
    }
    section_payloads = codec_payload.get("section_payloads")
    if isinstance(section_payloads, list) and section_payloads:
        for section_payload in section_payloads:
            counts[resolve_section_mode(section_payload)] += 1
        return counts

    for _section_range in codec_payload.get("section_ranges", []):
        counts[SINGLE_REFRESH_SECTION_MODE] += 1
    return counts


def count_effective_inference_steps(codec_payload: Dict[str, object], mode_inference_steps: Dict[str, int]) -> int:
    """根据 section mode 统计当前视频实际会执行的 denoise 步数。"""
    total_steps = 0
    section_payloads = codec_payload.get("section_payloads")
    if isinstance(section_payloads, list) and section_payloads:
        for section_payload in section_payloads:
            total_steps += int(mode_inference_steps[resolve_section_mode(section_payload)])
        return total_steps

    return len(codec_payload.get("section_ranges", [])) * int(mode_inference_steps[SINGLE_REFRESH_SECTION_MODE])


def build_section_anchor_blocks(
    low_full_latents: torch.Tensor,
    section_payload: Optional[Dict[str, object]],
    section_start: int,
    valid_target_frames: int,
    anchor_span_latents: int,
) -> List[Tuple[int, torch.Tensor]]:
    """从 low latent / sparse payload 里提取当前 section 的 anchor block 列表。"""
    anchor_blocks: List[Tuple[int, torch.Tensor]] = []
    if section_payload is not None:
        for anchor_block in section_payload.get("anchor_blocks", []):
            block_start = int(anchor_block.get("start", 0))
            block_length = int(anchor_block.get("length", 0))
            if block_length <= 0 or block_start >= valid_target_frames:
                continue
            block_end = min(valid_target_frames, block_start + block_length)
            source_start = section_start + block_start
            source_end = section_start + block_end
            anchor_values = low_full_latents[:, source_start:source_end]
            if anchor_values.shape[1] == 0:
                continue
            anchor_blocks.append((block_start, anchor_values[:, : block_end - block_start].contiguous()))
        if anchor_blocks:
            return anchor_blocks

    anchor_steps = min(anchor_span_latents, valid_target_frames)
    if anchor_steps <= 0:
        return anchor_blocks
    anchor_values = low_full_latents[:, section_start : section_start + anchor_steps]
    if anchor_values.shape[1] > 0:
        anchor_blocks.append((0, anchor_values.contiguous()))
    return anchor_blocks


def predict_section_from_history(
    history_latents: torch.Tensor,
    section_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
    """使用最后两帧历史做一个非常便宜的线性外推。"""
    channel_count, section_length, height, width = section_shape
    if section_length <= 0:
        raise ValueError(f"section_length must be > 0, got {section_length}.")
    if history_latents.ndim != 4:
        raise ValueError(
            f"Expected history_latents with shape [C, T, H, W], got {tuple(history_latents.shape)}."
        )
    if history_latents.shape[0] != channel_count:
        raise ValueError(
            f"History channel count {history_latents.shape[0]} does not match section channel count {channel_count}."
        )

    device = history_latents.device
    if history_latents.shape[1] == 0:
        return torch.zeros(section_shape, device=device, dtype=torch.float32)

    last_frame = history_latents[:, -1:].float()
    if history_latents.shape[1] >= 2:
        delta = (history_latents[:, -1:] - history_latents[:, -2:-1]).float()
    else:
        delta = torch.zeros_like(last_frame, dtype=torch.float32)

    steps = torch.linspace(
        1.0 / float(section_length),
        1.0,
        steps=section_length,
        device=device,
        dtype=torch.float32,
    ).view(1, section_length, 1, 1)
    predicted = last_frame.expand(-1, section_length, -1, -1) + steps * delta.expand(-1, section_length, -1, -1)
    if predicted.shape[2] != height or predicted.shape[3] != width:
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(0),
            size=(section_length, height, width),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)
    return predicted.float().contiguous()


def overlay_anchor_blocks(
    section_prediction: torch.Tensor,
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
    """把 anchor block 覆写到 section 预测上。"""
    updated = section_prediction.clone()
    for block_start, block_latents in anchor_blocks:
        if block_latents.ndim != 4:
            raise ValueError(
                f"Expected anchor block with shape [C, T, H, W], got {tuple(block_latents.shape)}."
            )
        block_length = int(block_latents.shape[1])
        if block_length <= 0:
            continue
        updated[:, block_start : block_start + block_length] = block_latents.to(
            device=updated.device,
            dtype=updated.dtype,
        )
    return updated.contiguous()


def interpolate_between_anchor_blocks(
    section_prediction: torch.Tensor,
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
    """当存在头尾双 anchor 时，对中间 gap 做线性插值。"""
    if len(anchor_blocks) < 2:
        return section_prediction

    updated = section_prediction.clone()
    sorted_blocks = sorted(anchor_blocks, key=lambda item: int(item[0]))
    for (left_start, left_block), (right_start, right_block) in zip(sorted_blocks[:-1], sorted_blocks[1:]):
        left_end = int(left_start) + int(left_block.shape[1]) - 1
        right_begin = int(right_start)
        gap = right_begin - left_end - 1
        if gap <= 0:
            continue
        left_frame = left_block[:, -1:].to(device=updated.device, dtype=updated.dtype)
        right_frame = right_block[:, :1].to(device=updated.device, dtype=updated.dtype)
        for offset in range(1, gap + 1):
            alpha = float(offset) / float(gap + 1)
            interpolated = (1.0 - alpha) * left_frame + alpha * right_frame
            updated[:, left_end + offset : left_end + offset + 1] = interpolated
    return updated.contiguous()


def build_proxy_section_from_history(
    history_latents: torch.Tensor,
    section_shape: Tuple[int, int, int, int],
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
    """为 0-step / 超低步数场景构造一个安全的 section 代理结果。"""
    prediction = predict_section_from_history(history_latents, section_shape)
    prediction = overlay_anchor_blocks(prediction, anchor_blocks)
    prediction = interpolate_between_anchor_blocks(prediction, anchor_blocks)
    prediction = overlay_anchor_blocks(prediction, anchor_blocks)
    return prediction.contiguous()


def build_section_condition_canvases_and_masks(
    low_full_latents: torch.Tensor,
    section_payload: Optional[Dict[str, object]],
    section_start: int,
    valid_target_frames: int,
    latent_window_size: int,
    anchor_span_latents: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hard_anchor_canvas = torch.zeros(
        low_full_latents.shape[0],
        latent_window_size,
        low_full_latents.shape[2],
        low_full_latents.shape[3],
        device=low_full_latents.device,
        dtype=low_full_latents.dtype,
    )
    hard_anchor_mask = torch.zeros(
        1,
        latent_window_size,
        1,
        1,
        device=low_full_latents.device,
        dtype=low_full_latents.dtype,
    )
    soft_tail_canvas = torch.zeros_like(hard_anchor_canvas)
    soft_tail_mask = torch.zeros_like(hard_anchor_mask)

    if section_payload is not None:
        for anchor_block in section_payload.get("anchor_blocks", []):
            block_start = int(anchor_block.get("start", 0))
            block_length = int(anchor_block.get("length", 0))
            if block_length <= 0 or block_start >= valid_target_frames:
                continue
            block_end = min(valid_target_frames, block_start + block_length)
            source_start = section_start + block_start
            source_end = section_start + block_end
            anchor_values = low_full_latents[:, source_start:source_end]
            if anchor_values.shape[1] == 0:
                continue
            head_codec_type = str(anchor_block.get("head_codec_type", DUAL_HEAD_ANCHOR_CODEC_TYPE))
            use_soft_condition = block_start != 0 or head_codec_type == DUAL_HEAD_SOFT_POOL_CODEC_TYPE
            target_canvas = soft_tail_canvas if use_soft_condition else hard_anchor_canvas
            target_mask = soft_tail_mask if use_soft_condition else hard_anchor_mask
            target_canvas[:, block_start:block_end] = anchor_values[:, : block_end - block_start]
            target_mask[:, block_start:block_end] = 1
        return (
            hard_anchor_canvas.contiguous(),
            hard_anchor_mask.contiguous(),
            soft_tail_canvas.contiguous(),
            soft_tail_mask.contiguous(),
        )

    anchor_steps = min(anchor_span_latents, valid_target_frames)
    if anchor_steps <= 0:
        return (
            hard_anchor_canvas.contiguous(),
            hard_anchor_mask.contiguous(),
            soft_tail_canvas.contiguous(),
            soft_tail_mask.contiguous(),
        )
    anchor_values = low_full_latents[:, section_start : section_start + anchor_steps]
    hard_anchor_canvas[:, :anchor_steps] = anchor_values
    hard_anchor_mask[:, :anchor_steps] = 1
    return (
        hard_anchor_canvas.contiguous(),
        hard_anchor_mask.contiguous(),
        soft_tail_canvas.contiguous(),
        soft_tail_mask.contiguous(),
    )


def build_section_anchor_canvas_and_mask(
    low_full_latents: torch.Tensor,
    section_payload: Optional[Dict[str, object]],
    section_start: int,
    valid_target_frames: int,
    latent_window_size: int,
    anchor_span_latents: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    hard_anchor_canvas, hard_anchor_mask, _soft_tail_canvas, _soft_tail_mask = (
        build_section_condition_canvases_and_masks(
            low_full_latents=low_full_latents,
            section_payload=section_payload,
            section_start=section_start,
            valid_target_frames=valid_target_frames,
            latent_window_size=latent_window_size,
            anchor_span_latents=anchor_span_latents,
        )
    )
    return hard_anchor_canvas, hard_anchor_mask


def blend_soft_condition_latents(
    latents: torch.Tensor,
    condition_canvas: torch.Tensor,
    condition_mask: torch.Tensor,
    strength: torch.Tensor | float,
) -> torch.Tensor:
    if latents.shape != condition_canvas.shape:
        raise ValueError(
            f"Latent shape {tuple(latents.shape)} does not match condition canvas shape {tuple(condition_canvas.shape)}."
        )
    if condition_mask.ndim != 5:
        raise ValueError(
            f"Expected condition_mask with shape [B, 1, T, 1, 1], got {tuple(condition_mask.shape)}."
        )
    if float(condition_mask.sum().item()) <= 0.0:
        return latents
    if torch.is_tensor(strength):
        strength_tensor = strength.to(device=latents.device, dtype=latents.dtype)
    else:
        strength_tensor = torch.tensor(float(strength), device=latents.device, dtype=latents.dtype)
    if float(strength_tensor.max().item()) <= 0.0:
        return latents
    while strength_tensor.ndim < latents.ndim:
        strength_tensor = strength_tensor.unsqueeze(-1)
    expanded_mask = condition_mask.to(device=latents.device, dtype=latents.dtype).expand_as(latents)
    condition_canvas = condition_canvas.to(device=latents.device, dtype=latents.dtype)
    blend_weight = (expanded_mask * strength_tensor).clamp_(0.0, 1.0)
    return (latents + (condition_canvas - latents) * blend_weight).contiguous()


def compute_soft_tail_hint_schedule_strength(
    *,
    step_idx: int,
    num_inference_steps: int,
    base_strength: float,
    step_fraction: float,
) -> float:
    if base_strength <= 0.0 or step_fraction <= 0.0 or num_inference_steps <= 0:
        return 0.0
    active_steps = max(1, int(math.ceil(float(num_inference_steps) * float(step_fraction))))
    if step_idx >= active_steps:
        return 0.0
    if active_steps == 1:
        return float(base_strength)
    remaining = 1.0 - float(step_idx) / float(active_steps - 1)
    return max(0.0, float(base_strength) * remaining)


def build_anchor_distance_weights(
    anchor_masks: torch.Tensor,
    valid_target_frames: torch.Tensor,
    latent_window_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = valid_target_frames.shape[0]
    weights = torch.zeros(batch_size, 1, latent_window_size, 1, 1, device=device, dtype=dtype)
    for idx, valid_frames in enumerate(valid_target_frames.tolist()):
        valid = int(valid_frames)
        if valid <= 0:
            continue
        anchor_positions = torch.nonzero(anchor_masks[idx, 0, :valid, 0, 0] > 0.5, as_tuple=False).flatten()
        if anchor_positions.numel() == 0:
            if valid == 1:
                weights[idx, 0, 0, 0, 0] = 1.0
            else:
                positions = torch.arange(valid, device=device, dtype=dtype)
                weights[idx, 0, :valid, 0, 0] = 1.0 + positions / float(valid - 1)
            continue

        anchor_positions = anchor_positions.to(device=device, dtype=dtype)
        positions = torch.arange(valid, device=device, dtype=dtype)
        distances = torch.abs(positions.view(-1, 1) - anchor_positions.view(1, -1)).min(dim=1).values
        non_anchor_mask = distances > 0
        if not bool(non_anchor_mask.any().item()):
            continue
        max_distance = float(distances[non_anchor_mask].max().item())
        if max_distance <= 0.0:
            max_distance = 1.0
        weights[idx, 0, :valid, 0, 0] = 1.0 + distances / max_distance
        weights[idx, 0, anchor_positions.to(dtype=torch.long), 0, 0] = 0.0
    return weights.contiguous()


def build_training_batch_tensors(
    batch: Dict[str, torch.Tensor | int],
    sequences: Sequence[PreparedSequence],
    codec_config: CodecConfig,
    learned_tail_codec: Optional[torch.nn.Module],
    device: torch.device,
    history_sizes: Sequence[int],
    latent_window_size: int,
    anchor_span_latents: int,
    train_mixed_dual_tail_spatial_factor: Optional[int] = None,
    train_mixed_dual_tail_ratio: float = 0.0,
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
    section_payload_cache: Dict[int, Dict[int, Dict[str, object]]] = {}
    for seq_idx in unique_seq_indices:
        sequence = sequences[seq_idx]
        codec_payload, low_full_latents = build_sequence_low_latents(
            sequence=sequence,
            codec_config=codec_config,
            learned_tail_codec=learned_tail_codec,
            device=device,
            use_ste_quant=codec_config.tail_codec_type == LEARNED_TAIL_CODEC_TYPE,
            move_to_cpu=False,
            train_mixed_dual_tail_spatial_factor=train_mixed_dual_tail_spatial_factor,
            train_mixed_dual_tail_ratio=train_mixed_dual_tail_ratio,
        )
        clean_latent_cache[seq_idx] = sequence.clean_full_latents.to(device=device, dtype=torch.float32).contiguous()
        low_latent_cache[seq_idx] = low_full_latents.to(device=device, dtype=torch.float32).contiguous()
        section_payload_cache[seq_idx] = build_section_payload_lookup(codec_payload)

    history_latents: List[torch.Tensor] = []
    target_latents: List[torch.Tensor] = []
    hard_anchor_canvases: List[torch.Tensor] = []
    hard_anchor_masks: List[torch.Tensor] = []
    soft_tail_canvases: List[torch.Tensor] = []
    soft_tail_masks: List[torch.Tensor] = []
    x0_latents: List[torch.Tensor] = []
    valid_target_frames: List[int] = []
    for seq_idx, section_start in zip(seq_indices, section_starts):
        clean_full_latents = clean_latent_cache[seq_idx]
        low_full_latents = low_latent_cache[seq_idx]
        section_payload = section_payload_cache[seq_idx].get(section_start)
        target_latent, valid_frames = extract_target_window(
            clean_full_latents=clean_full_latents,
            section_start=section_start,
            latent_window_size=latent_window_size,
        )
        history_latent = extract_history_window(
            source_full_latents=low_full_latents,
            section_start=section_start,
            history_window_size=history_window_size,
        )
        (
            hard_anchor_canvas,
            hard_anchor_mask,
            soft_tail_canvas,
            soft_tail_mask,
        ) = build_section_condition_canvases_and_masks(
            low_full_latents=low_full_latents,
            section_payload=section_payload,
            section_start=section_start,
            valid_target_frames=valid_frames,
            latent_window_size=latent_window_size,
            anchor_span_latents=anchor_span_latents,
        )
        history_latents.append(history_latent)
        target_latents.append(target_latent)
        hard_anchor_canvases.append(hard_anchor_canvas)
        hard_anchor_masks.append(hard_anchor_mask)
        soft_tail_canvases.append(soft_tail_canvas)
        soft_tail_masks.append(soft_tail_mask)
        x0_latents.append(low_full_latents[:, :1])
        valid_target_frames.append(valid_frames)

    return {
        "history_latents": torch.stack(history_latents, dim=0).contiguous(),
        "target_latents": torch.stack(target_latents, dim=0).contiguous(),
        "section_hard_anchor_canvas": torch.stack(hard_anchor_canvases, dim=0).contiguous(),
        "section_hard_anchor_mask": torch.stack(hard_anchor_masks, dim=0).contiguous(),
        "section_soft_tail_canvas": torch.stack(soft_tail_canvases, dim=0).contiguous(),
        "section_soft_tail_mask": torch.stack(soft_tail_masks, dim=0).contiguous(),
        "x0_latents": torch.stack(x0_latents, dim=0).contiguous(),
        "valid_target_frames": torch.tensor(valid_target_frames, device=device, dtype=torch.long),
    }


def load_transformer_bundle(
    base_model_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    checkpoint_dir: Optional[Path],
    gradient_checkpointing: bool,
    accelerator=None,
    freeze_after_load: Optional[bool] = None,
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
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
            local_files_only=True,
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            local_files_only=True,
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
            local_files_only=True,
        )
    transformer.to(device)
    scheduler = HeliosScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        local_files_only=True,
    )

    if freeze_after_load is None:
        freeze_after_load = checkpoint_dir is not None

    if checkpoint_dir is None:
        # 训练模式：从 base transformer 初始化，后续参数参与优化。
        transformer.requires_grad_(not freeze_after_load)
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
        transformer.requires_grad_(not freeze_after_load)

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
    aux_loss_clip_value: Optional[float],
    aux_loss_sigma_min: Optional[float],
    aux_loss_sigma_max: Optional[float],
    soft_tail_hint_strength: float,
    train_mixed_dual_tail_spatial_factor: Optional[int],
    train_mixed_dual_tail_ratio: float,
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
        train_mixed_dual_tail_spatial_factor=train_mixed_dual_tail_spatial_factor,
        train_mixed_dual_tail_ratio=train_mixed_dual_tail_ratio,
    )
    history_latents = materialized_batch["history_latents"].to(dtype=weight_dtype)
    target_latents = materialized_batch["target_latents"].to(dtype=weight_dtype)
    section_hard_anchor_canvas = materialized_batch["section_hard_anchor_canvas"].to(dtype=weight_dtype)
    section_hard_anchor_mask = materialized_batch["section_hard_anchor_mask"].to(dtype=weight_dtype)
    section_soft_tail_canvas = materialized_batch["section_soft_tail_canvas"].to(dtype=weight_dtype)
    section_soft_tail_mask = materialized_batch["section_soft_tail_mask"].to(dtype=weight_dtype)
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
    noisy_model_input = overwrite_anchor_latents(
        noisy_model_input,
        section_hard_anchor_canvas,
        section_hard_anchor_mask,
    )
    soft_tail_strength = sigma.to(dtype=weight_dtype).view(-1, 1, 1, 1, 1) * float(soft_tail_hint_strength)
    noisy_model_input = blend_soft_condition_latents(
        noisy_model_input,
        section_soft_tail_canvas,
        section_soft_tail_mask,
        soft_tail_strength,
    )
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
    valid_mask = build_valid_mask(
        valid_target_frames=valid_target_frames,
        latent_window_size=latent_window_size,
        device=device,
        dtype=torch.float32,
        valid_start=0,
    )
    hard_anchor_mask_float = section_hard_anchor_mask.to(device=device, dtype=torch.float32)
    mask = valid_mask * (1.0 - hard_anchor_mask_float)
    tail_position_weights = build_anchor_distance_weights(
        anchor_masks=hard_anchor_mask_float,
        valid_target_frames=valid_target_frames,
        latent_window_size=latent_window_size,
        device=device,
        dtype=torch.float32,
    )
    weighted_mask = mask * tail_position_weights
    aux_sigma_mask = torch.ones_like(sigma.view(-1, 1, 1, 1, 1), dtype=torch.float32)
    if aux_loss_sigma_min is not None:
        aux_sigma_mask = aux_sigma_mask * (sigma.view(-1, 1, 1, 1, 1) >= float(aux_loss_sigma_min)).float()
    if aux_loss_sigma_max is not None:
        aux_sigma_mask = aux_sigma_mask * (sigma.view(-1, 1, 1, 1, 1) <= float(aux_loss_sigma_max)).float()
    aux_weighted_mask = weighted_mask * aux_sigma_mask
    sigma_view_float = sigma.view(-1, 1, 1, 1, 1)
    xt = noisy_model_input.float()
    x_target = model_input.float()
    noise_target = noise.float()
    flow_target = flow_target.float()
    flow_pred = flow_pred.float()
    x_pred = xt - sigma_view_float * flow_pred
    noise_pred = xt + (1.0 - sigma_view_float) * flow_pred
    x_pred_for_temporal = overwrite_anchor_latents(
        x_pred.clone(),
        section_hard_anchor_canvas.float(),
        hard_anchor_mask_float,
    )
    x_target_for_temporal = overwrite_anchor_latents(
        x_target.clone(),
        section_hard_anchor_canvas.float(),
        hard_anchor_mask_float,
    )

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
    x_loss = normalized_masked_mse(x_pred, x_target, aux_weighted_mask)
    noise_loss = normalized_masked_mse(noise_pred, noise_target, aux_weighted_mask)
    temporal_delta_loss = compute_temporal_delta_loss(
        prediction=x_pred_for_temporal,
        target=x_target_for_temporal,
        valid_target_frames=valid_target_frames,
        device=device,
        position_weights=tail_position_weights * aux_sigma_mask,
    )
    x_loss_for_optim = clamp_loss_for_optimization(x_loss, aux_loss_clip_value)
    noise_loss_for_optim = clamp_loss_for_optimization(noise_loss, aux_loss_clip_value)
    temporal_delta_loss_for_optim = clamp_loss_for_optimization(temporal_delta_loss, aux_loss_clip_value)
    loss = (
        flow_loss_weight * flow_loss
        + aux_scale
        * (
            x_loss_weight * x_loss_for_optim
            + noise_loss_weight * noise_loss_for_optim
            + temporal_delta_loss_weight * temporal_delta_loss_for_optim
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


def clamp_loss_for_optimization(loss: torch.Tensor, clip_value: Optional[float]) -> torch.Tensor:
    if clip_value is None:
        return loss
    return loss.clamp(max=float(clip_value))


def overwrite_anchor_latents(
    latents: torch.Tensor,
    anchor_canvas: torch.Tensor,
    anchor_mask: torch.Tensor,
) -> torch.Tensor:
    if latents.shape != anchor_canvas.shape:
        raise ValueError(
            f"Latent shape {tuple(latents.shape)} does not match anchor canvas shape {tuple(anchor_canvas.shape)}."
        )
    if anchor_mask.ndim != 5:
        raise ValueError(f"Expected anchor_mask with shape [B, 1, T, 1, 1], got {tuple(anchor_mask.shape)}.")
    if float(anchor_mask.sum().item()) <= 0.0:
        return latents
    expanded_mask = anchor_mask.to(device=latents.device, dtype=latents.dtype).expand_as(latents)
    anchor_canvas = anchor_canvas.to(device=latents.device, dtype=latents.dtype)
    return (latents * (1.0 - expanded_mask) + anchor_canvas * expanded_mask).contiguous()


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
    mode_inference_steps: Dict[str, int],
    seed: int,
    fix_anchor_during_denoise: bool,
    soft_tail_hint_strength: float,
    soft_tail_hint_step_fraction: float,
    distributed_context: DistributedContext,
    timing: Optional[Dict[str, object]] = None,
) -> torch.Tensor:
    """把一段 low latents 重建成 recover latents。

    这里按 section 分段恢复，每个 section 只依赖：
    - 首帧 `x0`
    - 低码率历史窗口

    这与训练时的数据组织保持一致，也更贴近真实压缩恢复链路中的分段重建过程。
    """
    reconstruct_start_time = start_synced_timer(device) if timing is not None else None
    mode_order = (
        PREDICT_ONLY_SECTION_MODE,
        SINGLE_REFRESH_SECTION_MODE,
        DUAL_REFRESH_SECTION_MODE,
    )
    section_count_by_mode = {mode_name: 0 for mode_name in mode_order}
    section_seconds_by_mode = {mode_name: 0.0 for mode_name in mode_order}
    streaming_latent_ready_events: List[Dict[str, object]] = []
    denoise_submodule_seconds = {
        "total_seconds": 0.0,
        "random_init_and_scheduler_setup_seconds": 0.0,
        "anchor_overwrite_seconds": 0.0,
        "latent_to_weight_dtype_seconds": 0.0,
        "soft_tail_blend_seconds": 0.0,
        "transformer_forward_seconds": 0.0,
        "scheduler_step_seconds": 0.0,
        "finalize_seconds": 0.0,
        "num_steps": 0,
    }
    first_section_ready_seconds = None
    history_window_seconds = 0.0
    condition_build_seconds = 0.0
    stage1_prepare_inputs_seconds = 0.0
    proxy_section_build_seconds = 0.0
    stage1_denoise_seconds = 0.0
    prefix_concat_seconds = 0.0
    clean_full = sequence.clean_full_latents
    low_full = sequence.low_full_latents
    if clean_full.shape[1] == 1:
        if low_full is None:
            raise ValueError("sequence.low_full_latents is missing. Please materialize low latents first.")
        output = low_full[:, :1].clone().cpu().contiguous()
        if timing is not None:
            timing.clear()
            timing.update(
                {
                    "total_seconds": stop_synced_timer(reconstruct_start_time, device),
                    "first_section_ready_seconds": 0.0,
                    "history_window_seconds": 0.0,
                    "condition_build_seconds": 0.0,
                    "stage1_prepare_inputs_seconds": 0.0,
                    "proxy_section_build_seconds": 0.0,
                    "stage1_denoise_seconds": 0.0,
                    "prefix_concat_seconds": 0.0,
                    "assemble_output_seconds": 0.0,
                    "section_count": 0,
                    "section_count_by_mode": section_count_by_mode,
                    "section_seconds_by_mode": section_seconds_by_mode,
                    "avg_section_seconds_by_mode": {mode_name: 0.0 for mode_name in mode_order},
                    "streaming_latent_ready_events": [],
                    "stage1_denoise_submodule_seconds": denoise_submodule_seconds,
                }
            )
        return output
    if low_full is None or sequence.low_codec_payload is None:
        raise ValueError("sequence.low_full_latents / low_codec_payload are missing. Please materialize low latents first.")

    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    history_sizes = list(history_sizes)
    section_starts = list(range(1, clean_full.shape[1], latent_window_size))
    recovered_sections: List[torch.Tensor] = []
    recovered_prefix = low_full[:, :1].float().cpu().contiguous()
    low_history_source = low_full.float().cpu().contiguous()
    section_payload_lookup = build_section_payload_lookup(sequence.low_codec_payload)
    dummy_target = torch.zeros(
        1,
        clean_full.shape[0],
        latent_window_size,
        clean_full.shape[2],
        clean_full.shape[3],
        device=device,
        dtype=weight_dtype,
    )
    x0_latents = low_full[:, :1].unsqueeze(0).to(device=device, dtype=weight_dtype)
    validate_mode_inference_steps(mode_inference_steps)

    for section_idx, section_start in enumerate(
        tqdm(
            section_starts,
            desc=f"Reconstructing sections rank={distributed_context.rank}",
            disable=distributed_context.is_distributed and not distributed_context.is_main_process,
        )
    ):
        section_wall_start_time = start_synced_timer(device) if timing is not None else None
        history_start_time = start_synced_timer(device) if timing is not None else None
        history_latents = extract_history_window(
            source_full_latents=low_history_source,
            section_start=section_start,
            history_window_size=sum(history_sizes),
        ).unsqueeze(0)
        if history_start_time is not None:
            history_window_seconds += stop_synced_timer(history_start_time, device)
        valid_target_frames = min(latent_window_size, clean_full.shape[1] - section_start)
        section_payload = section_payload_lookup.get(section_start)
        section_mode = resolve_section_mode(section_payload)
        section_num_inference_steps = int(mode_inference_steps[section_mode])
        condition_start_time = start_synced_timer(device) if timing is not None else None
        (
            section_hard_anchor_canvas,
            section_hard_anchor_mask,
            section_soft_tail_canvas,
            section_soft_tail_mask,
        ) = build_section_condition_canvases_and_masks(
            low_full_latents=low_full,
            section_payload=section_payload,
            section_start=section_start,
            valid_target_frames=valid_target_frames,
            latent_window_size=latent_window_size,
            anchor_span_latents=anchor_span_latents,
        )
        if condition_start_time is not None:
            condition_build_seconds += stop_synced_timer(condition_start_time, device)

        if section_num_inference_steps == 0:
            proxy_start_time = start_synced_timer(device) if timing is not None else None
            anchor_blocks = build_section_anchor_blocks(
                low_full_latents=low_full,
                section_payload=section_payload,
                section_start=section_start,
                valid_target_frames=valid_target_frames,
                anchor_span_latents=anchor_span_latents,
            )
            valid_section = build_proxy_section_from_history(
                history_latents=history_latents[0].float().cpu(),
                section_shape=(
                    int(clean_full.shape[0]),
                    int(valid_target_frames),
                    int(clean_full.shape[2]),
                    int(clean_full.shape[3]),
                ),
                anchor_blocks=anchor_blocks,
            ).contiguous()
            if proxy_start_time is not None:
                proxy_section_build_seconds += stop_synced_timer(proxy_start_time, device)
        else:
            prepare_inputs_start_time = start_synced_timer(device) if timing is not None else None
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
            if prepare_inputs_start_time is not None:
                stage1_prepare_inputs_seconds += stop_synced_timer(prepare_inputs_start_time, device)
            section_denoise_timing: Dict[str, float] = {}
            denoise_start_time = start_synced_timer(device) if timing is not None else None
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
                num_inference_steps=section_num_inference_steps,
                seed=seed + section_start,
                hard_anchor_canvas=section_hard_anchor_canvas.unsqueeze(0).to(device=device, dtype=weight_dtype),
                hard_anchor_mask=section_hard_anchor_mask.unsqueeze(0).to(device=device, dtype=weight_dtype),
                soft_tail_canvas=section_soft_tail_canvas.unsqueeze(0).to(device=device, dtype=weight_dtype),
                soft_tail_mask=section_soft_tail_mask.unsqueeze(0).to(device=device, dtype=weight_dtype),
                fix_anchor_during_denoise=fix_anchor_during_denoise,
                soft_tail_hint_strength=soft_tail_hint_strength,
                soft_tail_hint_step_fraction=soft_tail_hint_step_fraction,
                timing=section_denoise_timing if timing is not None else None,
            )
            if denoise_start_time is not None:
                stage1_denoise_seconds += stop_synced_timer(denoise_start_time, device)
            for key, value in section_denoise_timing.items():
                if key == "num_steps":
                    denoise_submodule_seconds["num_steps"] += int(value)
                else:
                    denoise_submodule_seconds[key] += float(value)
            valid_section = section_latents[0, :, :valid_target_frames].contiguous()
        prefix_concat_start_time = start_synced_timer(device) if timing is not None else None
        recovered_sections.append(valid_section)
        recovered_prefix = torch.cat([recovered_prefix, valid_section.cpu()], dim=1).contiguous()
        if prefix_concat_start_time is not None:
            prefix_concat_seconds += stop_synced_timer(prefix_concat_start_time, device)
        section_count_by_mode[section_mode] += 1
        if section_wall_start_time is not None:
            section_wall_seconds = stop_synced_timer(section_wall_start_time, device)
            section_seconds_by_mode[section_mode] += float(section_wall_seconds)
            section_ready_seconds = stop_synced_timer(reconstruct_start_time, device)
            if first_section_ready_seconds is None:
                first_section_ready_seconds = section_ready_seconds
            for latent_offset in range(valid_target_frames):
                latent_index = int(section_start + latent_offset)
                frame_start = 1 + 4 * (latent_index - 1)
                frame_end = min(int(sequence.metadata.source_num_frames), frame_start + 4)
                if frame_end <= frame_start:
                    continue
                streaming_latent_ready_events.append(
                    {
                        "frame_group_index": len(streaming_latent_ready_events) + 1,
                        "latent_index": latent_index,
                        "section_index": int(section_idx),
                        "section_start_latent": int(section_start),
                        "section_mode": section_mode,
                        "frame_start": int(frame_start),
                        "frame_end": int(frame_end),
                        "ready_seconds": float(section_ready_seconds),
                        "section_elapsed_seconds": float(section_wall_seconds),
                    }
                )

    assemble_start_time = start_synced_timer(device) if timing is not None else None
    recovered_remainder = torch.cat(recovered_sections, dim=1)
    recovered_remainder = recovered_remainder[:, : clean_full.shape[1] - 1]
    output = torch.cat([low_full[:, :1].cpu(), recovered_remainder], dim=1).contiguous()
    if timing is not None:
        assemble_output_seconds = stop_synced_timer(assemble_start_time, device)
        timing.clear()
        timing.update(
            {
                "total_seconds": stop_synced_timer(reconstruct_start_time, device),
                "first_section_ready_seconds": first_section_ready_seconds,
                "history_window_seconds": history_window_seconds,
                "condition_build_seconds": condition_build_seconds,
                "stage1_prepare_inputs_seconds": stage1_prepare_inputs_seconds,
                "proxy_section_build_seconds": proxy_section_build_seconds,
                "stage1_denoise_seconds": stage1_denoise_seconds,
                "prefix_concat_seconds": prefix_concat_seconds,
                "assemble_output_seconds": assemble_output_seconds,
                "section_count": len(section_starts),
                "section_count_by_mode": section_count_by_mode,
                "section_seconds_by_mode": section_seconds_by_mode,
                "avg_section_seconds_by_mode": {
                    mode_name: (
                        section_seconds_by_mode[mode_name] / section_count_by_mode[mode_name]
                        if section_count_by_mode[mode_name] > 0
                        else 0.0
                    )
                    for mode_name in mode_order
                },
                "streaming_latent_ready_events": streaming_latent_ready_events,
                "stage1_denoise_submodule_seconds": denoise_submodule_seconds,
            }
        )
    return output


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
    hard_anchor_canvas: torch.Tensor,
    hard_anchor_mask: torch.Tensor,
    soft_tail_canvas: torch.Tensor,
    soft_tail_mask: torch.Tensor,
    fix_anchor_during_denoise: bool,
    soft_tail_hint_strength: float,
    soft_tail_hint_step_fraction: float,
    timing: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """执行单个 section 的扩散式去噪恢复。"""
    if num_inference_steps <= 0:
        raise ValueError(f"run_stage1_denoise requires num_inference_steps > 0, got {num_inference_steps}.")
    denoise_start_time = start_synced_timer(device) if timing is not None else None
    base_transformer = unwrap_model(transformer)
    generator = torch.Generator(device=device).manual_seed(seed)
    # 输入 hidden_states 是当前 section 的待恢复 latent，先随机初始化为高斯噪声；当前配置下 shape 近似是 [1, 16, 9, H, W]
    setup_start_time = start_synced_timer(device) if timing is not None else None
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32, generator=generator)
    anchor_overwrite_seconds = 0.0
    if fix_anchor_during_denoise:
        anchor_overwrite_start_time = start_synced_timer(device) if timing is not None else None
        latents = overwrite_anchor_latents(latents, hard_anchor_canvas.float(), hard_anchor_mask.float())
        if anchor_overwrite_start_time is not None:
            anchor_overwrite_seconds += stop_synced_timer(anchor_overwrite_start_time, device)
    prepare_stage1_inference_scheduler(
        scheduler=scheduler,
        latents=latents,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    setup_seconds = (
        stop_synced_timer(setup_start_time, device)
        if setup_start_time is not None
        else None
    )
    scheduler_type = str(get_scheduler_config_value(scheduler, "scheduler_type", "")).lower()
    supports_first_step_flag = model_supports_argument(base_transformer, "is_first_denoising_step")
    latent_to_weight_dtype_seconds = 0.0
    soft_tail_blend_seconds = 0.0
    transformer_forward_seconds = 0.0
    scheduler_step_seconds = 0.0

    for step_idx, timestep in enumerate(scheduler.timesteps):
        if fix_anchor_during_denoise:
            anchor_overwrite_start_time = start_synced_timer(device) if timing is not None else None
            latents = overwrite_anchor_latents(latents, hard_anchor_canvas.float(), hard_anchor_mask.float())
            if anchor_overwrite_start_time is not None:
                anchor_overwrite_seconds += stop_synced_timer(anchor_overwrite_start_time, device)
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
        cast_start_time = start_synced_timer(device) if timing is not None else None
        latent_model_input = latents.to(weight_dtype)
        if cast_start_time is not None:
            latent_to_weight_dtype_seconds += stop_synced_timer(cast_start_time, device)
        current_soft_tail_strength = compute_soft_tail_hint_schedule_strength(
            step_idx=step_idx,
            num_inference_steps=num_inference_steps,
            base_strength=soft_tail_hint_strength,
            step_fraction=soft_tail_hint_step_fraction,
        )
        soft_tail_start_time = start_synced_timer(device) if timing is not None else None
        latent_model_input = blend_soft_condition_latents(
            latent_model_input,
            soft_tail_canvas.to(weight_dtype),
            soft_tail_mask.to(weight_dtype),
            current_soft_tail_strength,
        )
        if soft_tail_start_time is not None:
            soft_tail_blend_seconds += stop_synced_timer(soft_tail_start_time, device)
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
            "return_dict": False,
        }
        if supports_first_step_flag:
            transformer_kwargs["is_first_denoising_step"] = step_idx == 0

        model_start_time = start_synced_timer(device) if timing is not None else None
        with get_model_cache_context(base_transformer, "cond"):
            # DiT的前向过程
            noise_pred = transformer(
                **transformer_kwargs,
            )[0]
        if model_start_time is not None:
            transformer_forward_seconds += stop_synced_timer(model_start_time, device)
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
        scheduler_step_start_time = start_synced_timer(device) if timing is not None else None
        if scheduler_type == "unipc" and hasattr(scheduler, "step_unipc"):
            latents = scheduler.step_unipc(noise_pred.float(), timestep, latents, return_dict=False)[0]
        else:
            latents = scheduler.step(noise_pred.float(), timestep, latents, return_dict=False)[0]
        if scheduler_step_start_time is not None:
            scheduler_step_seconds += stop_synced_timer(scheduler_step_start_time, device)
        if fix_anchor_during_denoise:
            anchor_overwrite_start_time = start_synced_timer(device) if timing is not None else None
            latents = overwrite_anchor_latents(latents, hard_anchor_canvas.float(), hard_anchor_mask.float())
            if anchor_overwrite_start_time is not None:
                anchor_overwrite_seconds += stop_synced_timer(anchor_overwrite_start_time, device)
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

    finalize_start_time = start_synced_timer(device) if timing is not None else None
    if hasattr(base_transformer, "clear_kv_cache"):
        # 长序列逐段推理后清 cache，避免显存逐步累积。
        base_transformer.clear_kv_cache()
    output = latents.cpu()
    finalize_seconds = (
        stop_synced_timer(finalize_start_time, device)
        if finalize_start_time is not None
        else None
    )
    if timing is not None:
        timing.clear()
        timing.update(
            {
                "total_seconds": stop_synced_timer(denoise_start_time, device),
                "random_init_and_scheduler_setup_seconds": 0.0 if setup_seconds is None else float(setup_seconds),
                "anchor_overwrite_seconds": anchor_overwrite_seconds,
                "latent_to_weight_dtype_seconds": latent_to_weight_dtype_seconds,
                "soft_tail_blend_seconds": soft_tail_blend_seconds,
                "transformer_forward_seconds": transformer_forward_seconds,
                "scheduler_step_seconds": scheduler_step_seconds,
                "finalize_seconds": 0.0 if finalize_seconds is None else float(finalize_seconds),
                "num_steps": len(scheduler.timesteps),
            }
        )
    return output


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
        "global_keyframe_codec_type": sequence.low_codec_payload.get(
            "global_keyframe_codec_type",
            sequence.low_codec_payload["codec_config"].get(
                "global_keyframe_codec_type",
                DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE,
            ),
        ),
        "global_keyframe": sequence.low_codec_payload.get("global_keyframe"),
        "global_keyframe_payload": sequence.low_codec_payload.get("global_keyframe_payload"),
        "section_ranges": [list(section_range) for section_range in sequence.low_codec_payload["section_ranges"]],
        "section_payloads": sequence.low_codec_payload.get("section_payloads"),
        "section_anchor_payloads": sequence.low_codec_payload.get("section_anchor_payloads"),
        "section_tail_payloads": sequence.low_codec_payload.get("section_tail_payloads"),
        "mode_selection_strategy": sequence.low_codec_payload.get("mode_selection_strategy"),
        "target_low_bpp": sequence.low_codec_payload.get("target_low_bpp"),
        "target_low_bytes": sequence.low_codec_payload.get("target_low_bytes"),
        "estimated_selected_bytes": sequence.low_codec_payload.get("estimated_selected_bytes"),
        "estimated_fixed_bytes": sequence.low_codec_payload.get("estimated_fixed_bytes"),
        "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
        "global_keyframe_bytes": int(sequence.low_codec_payload["global_keyframe_bytes"]),
        "single_head_bytes": int(sequence.low_codec_payload["single_head_bytes"]),
        "dual_head_bytes": int(sequence.low_codec_payload["dual_head_bytes"]),
        "dual_tail_presidual_bytes": int(sequence.low_codec_payload["dual_tail_presidual_bytes"]),
        "metadata_bytes": int(sequence.low_codec_payload["metadata_bytes"]),
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
            "global_keyframe_bytes": int(sequence.low_codec_payload["global_keyframe_bytes"]),
            "single_head_bytes": int(sequence.low_codec_payload["single_head_bytes"]),
            "dual_head_bytes": int(sequence.low_codec_payload["dual_head_bytes"]),
            "dual_tail_presidual_bytes": int(sequence.low_codec_payload["dual_tail_presidual_bytes"]),
            "metadata_bytes": int(sequence.low_codec_payload["metadata_bytes"]),
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
        "single_head_codec_type": codec_config.single_head_codec_type,
        "single_head_soft_pool_spatial_factor": codec_config.single_head_soft_pool_spatial_factor,
        "dual_head_codec_type": codec_config.dual_head_codec_type,
        "dual_head_anchor_spatial_factor": codec_config.dual_head_anchor_spatial_factor,
        "dual_head_p_delta_spatial_factor": codec_config.dual_head_p_delta_spatial_factor,
        "dual_head_soft_pool_spatial_factor": codec_config.dual_head_soft_pool_spatial_factor,
        "adaptive_dual_head_full_ratio": codec_config.adaptive_dual_head_full_ratio,
        "adaptive_dual_head_soft_pool_hard_factor": codec_config.adaptive_dual_head_soft_pool_hard_factor,
        "adaptive_dual_head_soft_pool_hard_ratio": codec_config.adaptive_dual_head_soft_pool_hard_ratio,
        "global_keyframe_codec_type": codec_config.global_keyframe_codec_type,
        "global_keyframe_quant_dtype": codec_config.global_keyframe_quant_dtype,
        "global_keyframe_spatial_factor": codec_config.global_keyframe_spatial_factor,
        "dual_tail_anchor_spatial_factor": codec_config.dual_tail_anchor_spatial_factor,
        "dual_tail_codec_type": codec_config.dual_tail_codec_type,
        "dual_tail_p_delta_spatial_factor": codec_config.dual_tail_p_delta_spatial_factor,
        "tail_codec_type": codec_config.tail_codec_type,
        "num_inference_steps": args.num_inference_steps,
        "low_latent_format_version": DEFAULT_LOW_LATENT_FORMAT_VERSION,
        "learned_codec_checkpoint": LEARNED_CODEC_CHECKPOINT_NAME if learned_tail_codec is not None else None,
        "learned_codec_config": learned_codec_config,
        "fix_anchor_during_denoise": args.fix_anchor_during_denoise,
        "soft_tail_hint_strength": args.soft_tail_hint_strength,
        "soft_tail_hint_step_fraction": args.soft_tail_hint_step_fraction,
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
            "aux_loss_clip_value": args.aux_loss_clip_value,
            "aux_loss_sigma_min": args.aux_loss_sigma_min,
            "aux_loss_sigma_max": args.aux_loss_sigma_max,
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

    required_keys = {"format_version", "history_sizes", "latent_window_size", "codec_config"}
    missing_keys = sorted(required_keys - config.keys())
    if missing_keys:
        raise KeyError(f"Recover config {config_path} is missing keys: {missing_keys}")

    format_version = config["format_version"]
    if format_version not in {"helios_recover_v2", "helios_recover_v3", RECOVER_CONFIG_VERSION}:
        raise ValueError(
            f"Unsupported recover config format in {config_path}: {format_version}. "
            f"Expected helios_recover_v2, helios_recover_v3, or {RECOVER_CONFIG_VERSION}."
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
    codec_config.setdefault(
        "single_head_codec_type",
        str(config.get("single_head_codec_type", DEFAULT_SINGLE_HEAD_CODEC_TYPE)),
    )
    codec_config.setdefault(
        "single_head_soft_pool_spatial_factor",
        int(
            config.get(
                "single_head_soft_pool_spatial_factor",
                DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
            )
        ),
    )
    dual_head_anchor_spatial_factor = config.get("dual_head_anchor_spatial_factor")
    codec_config.setdefault(
        "dual_head_codec_type",
        str(config.get("dual_head_codec_type", DEFAULT_DUAL_HEAD_CODEC_TYPE)),
    )
    codec_config.setdefault(
        "dual_head_anchor_spatial_factor",
        None if dual_head_anchor_spatial_factor is None else int(dual_head_anchor_spatial_factor),
    )
    codec_config.setdefault(
        "dual_head_p_delta_spatial_factor",
        int(config.get("dual_head_p_delta_spatial_factor", DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR)),
    )
    codec_config.setdefault(
        "dual_head_soft_pool_spatial_factor",
        int(config.get("dual_head_soft_pool_spatial_factor", DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR)),
    )
    codec_config.setdefault(
        "adaptive_dual_head_full_ratio",
        float(config.get("adaptive_dual_head_full_ratio", DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO)),
    )
    adaptive_dual_head_soft_pool_hard_factor = config.get("adaptive_dual_head_soft_pool_hard_factor")
    codec_config.setdefault(
        "adaptive_dual_head_soft_pool_hard_factor",
        None
        if adaptive_dual_head_soft_pool_hard_factor is None
        else int(adaptive_dual_head_soft_pool_hard_factor),
    )
    codec_config.setdefault(
        "adaptive_dual_head_soft_pool_hard_ratio",
        float(
            config.get(
                "adaptive_dual_head_soft_pool_hard_ratio",
                DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
            )
        ),
    )
    codec_config.setdefault(
        "global_keyframe_codec_type",
        str(config.get("global_keyframe_codec_type", DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE)),
    )
    codec_config.setdefault(
        "global_keyframe_quant_dtype",
        str(config.get("global_keyframe_quant_dtype", DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE)),
    )
    codec_config.setdefault(
        "global_keyframe_spatial_factor",
        int(config.get("global_keyframe_spatial_factor", DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR)),
    )
    dual_tail_anchor_spatial_factor = config.get("dual_tail_anchor_spatial_factor")
    codec_config.setdefault(
        "dual_tail_anchor_spatial_factor",
        None if dual_tail_anchor_spatial_factor is None else int(dual_tail_anchor_spatial_factor),
    )
    codec_config.setdefault(
        "dual_tail_codec_type",
        str(config.get("dual_tail_codec_type", DEFAULT_DUAL_TAIL_CODEC_TYPE)),
    )
    codec_config.setdefault(
        "dual_tail_p_delta_spatial_factor",
        int(config.get("dual_tail_p_delta_spatial_factor", DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR)),
    )
    codec_config.setdefault("tail_codec_type", config.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    learned_codec_config = dict(config.get("learned_codec_config") or {})
    if "hidden_channels" in learned_codec_config and "learned_codec_hidden_channels" not in codec_config:
        codec_config["learned_codec_hidden_channels"] = int(learned_codec_config["hidden_channels"])

    config["format_version"] = RECOVER_CONFIG_VERSION
    config["codec_config"] = codec_config
    config["section_span_latents"] = int(codec_config["section_span_latents"])
    config["anchor_span_latents"] = int(codec_config["anchor_span_latents"])
    config["tail_span_latents"] = int(codec_config["tail_span_latents"])
    config["anchor_quant_dtype"] = codec_config["anchor_quant_dtype"]
    config["anchor_spatial_factor"] = int(codec_config["anchor_spatial_factor"])
    config["single_head_codec_type"] = str(codec_config["single_head_codec_type"])
    config["single_head_soft_pool_spatial_factor"] = int(
        codec_config["single_head_soft_pool_spatial_factor"]
    )
    config["dual_head_codec_type"] = str(codec_config["dual_head_codec_type"])
    config["dual_head_anchor_spatial_factor"] = codec_config["dual_head_anchor_spatial_factor"]
    config["dual_head_p_delta_spatial_factor"] = int(codec_config["dual_head_p_delta_spatial_factor"])
    config["dual_head_soft_pool_spatial_factor"] = int(codec_config["dual_head_soft_pool_spatial_factor"])
    config["adaptive_dual_head_full_ratio"] = float(codec_config["adaptive_dual_head_full_ratio"])
    config["adaptive_dual_head_soft_pool_hard_factor"] = codec_config[
        "adaptive_dual_head_soft_pool_hard_factor"
    ]
    config["adaptive_dual_head_soft_pool_hard_ratio"] = float(
        codec_config["adaptive_dual_head_soft_pool_hard_ratio"]
    )
    config["global_keyframe_codec_type"] = str(codec_config["global_keyframe_codec_type"])
    config["global_keyframe_quant_dtype"] = str(codec_config["global_keyframe_quant_dtype"])
    config["global_keyframe_spatial_factor"] = int(codec_config["global_keyframe_spatial_factor"])
    config["dual_tail_anchor_spatial_factor"] = codec_config["dual_tail_anchor_spatial_factor"]
    config["dual_tail_codec_type"] = str(codec_config["dual_tail_codec_type"])
    config["dual_tail_p_delta_spatial_factor"] = int(codec_config["dual_tail_p_delta_spatial_factor"])
    config["tail_codec_type"] = str(codec_config["tail_codec_type"])
    config["num_inference_steps"] = int(config.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS))
    config["low_latent_format_version"] = config.get(
        "low_latent_format_version",
        DEFAULT_LOW_LATENT_FORMAT_VERSION,
    )
    config["learned_codec_checkpoint"] = config.get("learned_codec_checkpoint")
    config["learned_codec_config"] = learned_codec_config
    config.setdefault("fix_anchor_during_denoise", DEFAULT_FIX_ANCHOR_DURING_DENOISE)
    config.setdefault("soft_tail_hint_strength", DEFAULT_SOFT_TAIL_HINT_STRENGTH)
    config.setdefault("soft_tail_hint_step_fraction", DEFAULT_SOFT_TAIL_HINT_STEP_FRACTION)
    train_loss_config = dict(config.get("train_loss_config") or {})
    train_loss_config.setdefault("temporal_delta_loss_weight", DEFAULT_TEMPORAL_DELTA_LOSS_WEIGHT)
    train_loss_config.setdefault("aux_loss_clip_value", DEFAULT_AUX_LOSS_CLIP_VALUE)
    train_loss_config.setdefault("aux_loss_sigma_min", DEFAULT_AUX_LOSS_SIGMA_MIN)
    train_loss_config.setdefault("aux_loss_sigma_max", DEFAULT_AUX_LOSS_SIGMA_MAX)
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
    if cli_value is None:
        return checkpoint_value if checkpoint_value is not None else default_value
    if checkpoint_value is None:
        return cli_value
    if cli_value != default_value:
        return cli_value
    return checkpoint_value


def resolve_optional_infer_int(cli_value: Optional[int], checkpoint_value: object, fallback_value: int) -> int:
    """解析可选的整型推理覆盖值。"""
    if checkpoint_value is None:
        checkpoint_int = None
    else:
        checkpoint_int = int(checkpoint_value)
    resolved = cli_value if cli_value is not None else checkpoint_int
    if resolved is None:
        resolved = int(fallback_value)
    return int(resolved)


def resolve_mode_inference_steps(args: argparse.Namespace, checkpoint_config: Dict[str, object]) -> Dict[str, int]:
    """解析不同 section mode 在推理时使用的 denoise 步数。"""
    global_steps = int(args.num_inference_steps)
    mode_steps = {
        PREDICT_ONLY_SECTION_MODE: resolve_optional_infer_int(
            cli_value=args.predict_only_steps,
            checkpoint_value=checkpoint_config.get("predict_only_num_inference_steps"),
            fallback_value=global_steps,
        ),
        SINGLE_REFRESH_SECTION_MODE: resolve_optional_infer_int(
            cli_value=args.single_refresh_steps,
            checkpoint_value=checkpoint_config.get("single_refresh_num_inference_steps"),
            fallback_value=global_steps,
        ),
        DUAL_REFRESH_SECTION_MODE: resolve_optional_infer_int(
            cli_value=args.dual_refresh_steps,
            checkpoint_value=checkpoint_config.get("dual_refresh_num_inference_steps"),
            fallback_value=global_steps,
        ),
    }
    validate_mode_inference_steps(mode_steps)
    return mode_steps


def validate_mode_inference_steps(mode_inference_steps: Dict[str, int]) -> None:
    """校验按 mode 配置的推理步数。"""
    missing_modes = sorted(VALID_SECTION_MODES - set(mode_inference_steps.keys()))
    if missing_modes:
        raise KeyError(f"mode_inference_steps is missing modes: {missing_modes}")
    for mode_name, step_count in mode_inference_steps.items():
        if int(step_count) < 0:
            raise ValueError(f"Inference steps for mode={mode_name} must be >= 0, got {step_count}.")


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
