import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from reconstruct.learned_codec import LEARNED_TAIL_CODEC_TYPE


EPS = 1e-8
TRILINEAR_TAIL_CODEC_TYPE = "trilinear"
RAW_GLOBAL_KEYFRAME_CODEC_TYPE = "raw_fp16"
INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE = "int8_refresh_like"
DUAL_HEAD_ANCHOR_CODEC_TYPE = "anchor"
DUAL_HEAD_P_DELTA_CODEC_TYPE = "p_delta"
DUAL_HEAD_SOFT_POOL_CODEC_TYPE = "soft_pool"
SINGLE_HEAD_ANCHOR_CODEC_TYPE = DUAL_HEAD_ANCHOR_CODEC_TYPE
SINGLE_HEAD_SOFT_POOL_CODEC_TYPE = DUAL_HEAD_SOFT_POOL_CODEC_TYPE
DUAL_TAIL_ANCHOR_CODEC_TYPE = "anchor"
DUAL_TAIL_P_DELTA_CODEC_TYPE = "p_delta"
PREDICT_ONLY_SECTION_MODE = "predict_only"
SINGLE_REFRESH_SECTION_MODE = "single_refresh"
DUAL_REFRESH_SECTION_MODE = "dual_refresh"
DEFAULT_MAX_PREDICT_ONLY_GAP_SECTIONS = 2
DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD = 0.12
DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD = 0.18
DEFAULT_BOUNDARY_JUMP_THRESHOLD = 0.18
DEFAULT_CUT_DETECTION_THRESHOLD = 0.35
DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE = RAW_GLOBAL_KEYFRAME_CODEC_TYPE
DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE = "int8"
DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR = 1
DEFAULT_SINGLE_HEAD_CODEC_TYPE = SINGLE_HEAD_ANCHOR_CODEC_TYPE
DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR = 4
DEFAULT_DUAL_HEAD_CODEC_TYPE = DUAL_HEAD_ANCHOR_CODEC_TYPE
DEFAULT_DUAL_HEAD_ANCHOR_SPATIAL_FACTOR: Optional[int] = None
DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR = 4
DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR = 4
DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO = 0.0
DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_FACTOR: Optional[int] = None
DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO = 0.0
DEFAULT_DUAL_TAIL_CODEC_TYPE = DUAL_TAIL_ANCHOR_CODEC_TYPE
DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR = 4
LOW_CODEC_BYTE_BREAKDOWN_KEYS = (
    "global_keyframe_bytes",
    "single_head_bytes",
    "dual_head_bytes",
    "dual_tail_presidual_bytes",
    "metadata_bytes",
)


def _get_config_value(codec_config, key: str):
    if isinstance(codec_config, dict):
        return codec_config[key]
    return getattr(codec_config, key)


def _get_optional_config_value(codec_config, key: str, default):
    if isinstance(codec_config, dict):
        return codec_config.get(key, default)
    return getattr(codec_config, key, default)


def _as_int(codec_config, key: str) -> int:
    return int(_get_config_value(codec_config, key))


def _as_str(codec_config, key: str) -> str:
    return str(_get_config_value(codec_config, key))


def _as_float(codec_config, key: str, default: float) -> float:
    return float(_get_optional_config_value(codec_config, key, default))


def _as_optional_float(codec_config, key: str) -> Optional[float]:
    value = _get_optional_config_value(codec_config, key, None)
    if value is None:
        return None
    return float(value)


def _tail_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    return str(getattr(codec_config, "tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))


def _global_keyframe_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("global_keyframe_codec_type", DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE))
    return str(getattr(codec_config, "global_keyframe_codec_type", DEFAULT_GLOBAL_KEYFRAME_CODEC_TYPE))


def _global_keyframe_quant_dtype(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("global_keyframe_quant_dtype", DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE))
    return str(getattr(codec_config, "global_keyframe_quant_dtype", DEFAULT_GLOBAL_KEYFRAME_QUANT_DTYPE))


def _global_keyframe_spatial_factor(codec_config) -> int:
    if isinstance(codec_config, dict):
        return int(codec_config.get("global_keyframe_spatial_factor", DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR))
    return int(getattr(codec_config, "global_keyframe_spatial_factor", DEFAULT_GLOBAL_KEYFRAME_SPATIAL_FACTOR))


def _dual_tail_anchor_spatial_factor(codec_config) -> Optional[int]:
    value = _get_optional_config_value(codec_config, "dual_tail_anchor_spatial_factor", None)
    if value is None:
        return None
    return int(value)


def _dual_head_anchor_spatial_factor(codec_config) -> Optional[int]:
    value = _get_optional_config_value(codec_config, "dual_head_anchor_spatial_factor", None)
    if value is None:
        return None
    return int(value)


def _single_head_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("single_head_codec_type", DEFAULT_SINGLE_HEAD_CODEC_TYPE))
    return str(getattr(codec_config, "single_head_codec_type", DEFAULT_SINGLE_HEAD_CODEC_TYPE))


def _single_head_soft_pool_spatial_factor(codec_config) -> int:
    if isinstance(codec_config, dict):
        return int(
            codec_config.get(
                "single_head_soft_pool_spatial_factor",
                DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
            )
        )
    return int(
        getattr(
            codec_config,
            "single_head_soft_pool_spatial_factor",
            DEFAULT_SINGLE_HEAD_SOFT_POOL_SPATIAL_FACTOR,
        )
    )


def _dual_head_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("dual_head_codec_type", DEFAULT_DUAL_HEAD_CODEC_TYPE))
    return str(getattr(codec_config, "dual_head_codec_type", DEFAULT_DUAL_HEAD_CODEC_TYPE))


def _dual_head_p_delta_spatial_factor(codec_config) -> int:
    if isinstance(codec_config, dict):
        return int(codec_config.get("dual_head_p_delta_spatial_factor", DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR))
    return int(getattr(codec_config, "dual_head_p_delta_spatial_factor", DEFAULT_DUAL_HEAD_P_DELTA_SPATIAL_FACTOR))


def _dual_head_soft_pool_spatial_factor(codec_config) -> int:
    if isinstance(codec_config, dict):
        return int(codec_config.get("dual_head_soft_pool_spatial_factor", DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR))
    return int(getattr(codec_config, "dual_head_soft_pool_spatial_factor", DEFAULT_DUAL_HEAD_SOFT_POOL_SPATIAL_FACTOR))


def _adaptive_dual_head_full_ratio(codec_config) -> float:
    if isinstance(codec_config, dict):
        return float(codec_config.get("adaptive_dual_head_full_ratio", DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO))
    return float(getattr(codec_config, "adaptive_dual_head_full_ratio", DEFAULT_ADAPTIVE_DUAL_HEAD_FULL_RATIO))


def _adaptive_dual_head_soft_pool_hard_factor(codec_config) -> Optional[int]:
    value = _get_optional_config_value(codec_config, "adaptive_dual_head_soft_pool_hard_factor", None)
    if value is None:
        return None
    return int(value)


def _adaptive_dual_head_soft_pool_hard_ratio(codec_config) -> float:
    if isinstance(codec_config, dict):
        return float(
            codec_config.get(
                "adaptive_dual_head_soft_pool_hard_ratio",
                DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
            )
        )
    return float(
        getattr(
            codec_config,
            "adaptive_dual_head_soft_pool_hard_ratio",
            DEFAULT_ADAPTIVE_DUAL_HEAD_SOFT_POOL_HARD_RATIO,
        )
    )


def _dual_tail_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("dual_tail_codec_type", DEFAULT_DUAL_TAIL_CODEC_TYPE))
    return str(getattr(codec_config, "dual_tail_codec_type", DEFAULT_DUAL_TAIL_CODEC_TYPE))


def _dual_tail_p_delta_spatial_factor(codec_config) -> int:
    if isinstance(codec_config, dict):
        return int(codec_config.get("dual_tail_p_delta_spatial_factor", DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR))
    return int(getattr(codec_config, "dual_tail_p_delta_spatial_factor", DEFAULT_DUAL_TAIL_P_DELTA_SPATIAL_FACTOR))


def _maybe_to_cpu(tensor: torch.Tensor, move_to_cpu: bool) -> torch.Tensor:
    if move_to_cpu:
        return tensor.cpu().contiguous()
    return tensor.contiguous()


def _module_device(module) -> Optional[torch.device]:
    if module is None:
        return None
    try:
        return next(module.parameters()).device
    except StopIteration:
        return None


def validate_codec_config(codec_config) -> None:
    section_span_latents = _as_int(codec_config, "section_span_latents")
    anchor_span_latents = _as_int(codec_config, "anchor_span_latents")
    tail_span_latents = _as_int(codec_config, "tail_span_latents")
    temporal_factor = _as_int(codec_config, "temporal_factor")
    spatial_factor = _as_int(codec_config, "spatial_factor")
    anchor_spatial_factor = _as_int(codec_config, "anchor_spatial_factor")
    dual_tail_anchor_spatial_factor = _dual_tail_anchor_spatial_factor(codec_config)
    single_head_codec_type = _single_head_codec_type(codec_config)
    single_head_soft_pool_spatial_factor = _single_head_soft_pool_spatial_factor(codec_config)
    dual_head_codec_type = _dual_head_codec_type(codec_config)
    dual_head_anchor_spatial_factor = _dual_head_anchor_spatial_factor(codec_config)
    dual_head_p_delta_spatial_factor = _dual_head_p_delta_spatial_factor(codec_config)
    dual_head_soft_pool_spatial_factor = _dual_head_soft_pool_spatial_factor(codec_config)
    adaptive_dual_head_full_ratio = _adaptive_dual_head_full_ratio(codec_config)
    adaptive_dual_head_soft_pool_hard_factor = _adaptive_dual_head_soft_pool_hard_factor(codec_config)
    adaptive_dual_head_soft_pool_hard_ratio = _adaptive_dual_head_soft_pool_hard_ratio(codec_config)
    dual_tail_codec_type = _dual_tail_codec_type(codec_config)
    dual_tail_p_delta_spatial_factor = _dual_tail_p_delta_spatial_factor(codec_config)

    if section_span_latents < 1:
        raise ValueError(f"section_span_latents must be >= 1, got {section_span_latents}.")
    if anchor_span_latents < 1:
        raise ValueError(f"anchor_span_latents must be >= 1, got {anchor_span_latents}.")
    if tail_span_latents < 0:
        raise ValueError(f"tail_span_latents must be >= 0, got {tail_span_latents}.")
    if anchor_span_latents + tail_span_latents != section_span_latents:
        raise ValueError(
            "anchor_span_latents + tail_span_latents must equal section_span_latents, "
            f"got {anchor_span_latents} + {tail_span_latents} != {section_span_latents}."
        )
    if temporal_factor < 1 or spatial_factor < 1 or anchor_spatial_factor < 1:
        raise ValueError(
            "temporal_factor, spatial_factor, and anchor_spatial_factor must all be >= 1, "
            f"got temporal_factor={temporal_factor}, spatial_factor={spatial_factor}, "
            f"anchor_spatial_factor={anchor_spatial_factor}."
        )
    global_keyframe_spatial_factor = _global_keyframe_spatial_factor(codec_config)
    if global_keyframe_spatial_factor < 1:
        raise ValueError(
            "global_keyframe_spatial_factor must be >= 1, "
            f"got {global_keyframe_spatial_factor}."
        )
    if dual_tail_anchor_spatial_factor is not None and dual_tail_anchor_spatial_factor < 1:
        raise ValueError(
            "dual_tail_anchor_spatial_factor must be >= 1 when set, "
            f"got {dual_tail_anchor_spatial_factor}."
        )
    if dual_head_anchor_spatial_factor is not None and dual_head_anchor_spatial_factor < 1:
        raise ValueError(
            "dual_head_anchor_spatial_factor must be >= 1 when set, "
            f"got {dual_head_anchor_spatial_factor}."
        )
    if single_head_codec_type not in {SINGLE_HEAD_ANCHOR_CODEC_TYPE, SINGLE_HEAD_SOFT_POOL_CODEC_TYPE}:
        raise ValueError(
            f"Unsupported single_head_codec_type={single_head_codec_type}. "
            f"Expected {SINGLE_HEAD_ANCHOR_CODEC_TYPE} or {SINGLE_HEAD_SOFT_POOL_CODEC_TYPE}."
        )
    if single_head_soft_pool_spatial_factor < 1:
        raise ValueError(
            "single_head_soft_pool_spatial_factor must be >= 1, "
            f"got {single_head_soft_pool_spatial_factor}."
        )
    if dual_head_codec_type not in {
        DUAL_HEAD_ANCHOR_CODEC_TYPE,
        DUAL_HEAD_P_DELTA_CODEC_TYPE,
        DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
    }:
        raise ValueError(
            f"Unsupported dual_head_codec_type={dual_head_codec_type}. "
            f"Expected {DUAL_HEAD_ANCHOR_CODEC_TYPE}, {DUAL_HEAD_P_DELTA_CODEC_TYPE}, "
            f"or {DUAL_HEAD_SOFT_POOL_CODEC_TYPE}."
        )
    if dual_head_p_delta_spatial_factor < 1:
        raise ValueError(
            "dual_head_p_delta_spatial_factor must be >= 1, "
            f"got {dual_head_p_delta_spatial_factor}."
        )
    if dual_head_soft_pool_spatial_factor < 1:
        raise ValueError(
            "dual_head_soft_pool_spatial_factor must be >= 1, "
            f"got {dual_head_soft_pool_spatial_factor}."
        )
    if not 0.0 <= adaptive_dual_head_full_ratio <= 1.0:
        raise ValueError(
            "adaptive_dual_head_full_ratio must be in [0, 1], "
            f"got {adaptive_dual_head_full_ratio}."
        )
    if (
        adaptive_dual_head_soft_pool_hard_factor is not None
        and adaptive_dual_head_soft_pool_hard_factor < 1
    ):
        raise ValueError(
            "adaptive_dual_head_soft_pool_hard_factor must be >= 1 when set, "
            f"got {adaptive_dual_head_soft_pool_hard_factor}."
        )
    if not 0.0 <= adaptive_dual_head_soft_pool_hard_ratio <= 1.0:
        raise ValueError(
            "adaptive_dual_head_soft_pool_hard_ratio must be in [0, 1], "
            f"got {adaptive_dual_head_soft_pool_hard_ratio}."
        )
    if dual_tail_codec_type not in {DUAL_TAIL_ANCHOR_CODEC_TYPE, DUAL_TAIL_P_DELTA_CODEC_TYPE}:
        raise ValueError(
            f"Unsupported dual_tail_codec_type={dual_tail_codec_type}. "
            f"Expected {DUAL_TAIL_ANCHOR_CODEC_TYPE} or {DUAL_TAIL_P_DELTA_CODEC_TYPE}."
        )
    if dual_tail_p_delta_spatial_factor < 1:
        raise ValueError(
            "dual_tail_p_delta_spatial_factor must be >= 1, "
            f"got {dual_tail_p_delta_spatial_factor}."
        )
    for key in ("quant_dtype", "anchor_quant_dtype"):
        value = _as_str(codec_config, key)
        if value != "int8":
            raise ValueError(f"{key}={value} is unsupported. Only int8 is implemented.")
    keyframe_dtype = _as_str(codec_config, "keyframe_dtype")
    if keyframe_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported keyframe_dtype={keyframe_dtype}.")
    global_keyframe_codec_type = _global_keyframe_codec_type(codec_config)
    if global_keyframe_codec_type not in {
        RAW_GLOBAL_KEYFRAME_CODEC_TYPE,
        INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE,
    }:
        raise ValueError(
            f"Unsupported global_keyframe_codec_type={global_keyframe_codec_type}. "
            f"Expected {RAW_GLOBAL_KEYFRAME_CODEC_TYPE} or {INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE}."
        )
    global_keyframe_quant_dtype = _global_keyframe_quant_dtype(codec_config)
    if global_keyframe_quant_dtype != "int8":
        raise ValueError(
            "global_keyframe_quant_dtype="
            f"{global_keyframe_quant_dtype} is unsupported. Only int8 is implemented."
        )
    tail_codec_type = _tail_codec_type(codec_config)
    if tail_codec_type not in {TRILINEAR_TAIL_CODEC_TYPE, LEARNED_TAIL_CODEC_TYPE}:
        raise ValueError(
            f"Unsupported tail_codec_type={tail_codec_type}. "
            f"Expected {TRILINEAR_TAIL_CODEC_TYPE} or {LEARNED_TAIL_CODEC_TYPE}."
        )
    target_low_bpp = _as_optional_float(codec_config, "target_low_bpp")
    if target_low_bpp is not None and target_low_bpp <= 0.0:
        raise ValueError(f"target_low_bpp must be > 0 when set, got {target_low_bpp}.")


def make_section_ranges(
    total_latent_frames: int,
    section_span_latents: int,
    start_index: int = 1,
) -> List[Tuple[int, int]]:
    if total_latent_frames < 0:
        raise ValueError(f"total_latent_frames must be >= 0, got {total_latent_frames}.")
    if section_span_latents < 1:
        raise ValueError(f"section_span_latents must be >= 1, got {section_span_latents}.")
    if start_index < 0:
        raise ValueError(f"start_index must be >= 0, got {start_index}.")

    ranges: List[Tuple[int, int]] = []
    for start in range(start_index, total_latent_frames, section_span_latents):
        end = min(total_latent_frames, start + section_span_latents)
        ranges.append((start, end))
    return ranges


def trilinear_resize(latents: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(latents.float(), size=size, mode="trilinear", align_corners=False)


def symmetric_quantize_per_channel(latents: torch.Tensor, quant_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if quant_dtype != "int8":
        raise ValueError(f"Unsupported quant_dtype={quant_dtype}. Only int8 is implemented.")
    scales = latents.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(EPS) / 127.0
    quantized = torch.round(latents / scales).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.squeeze(-1).squeeze(-1).squeeze(-1).float().contiguous()


def symmetric_dequantize_per_channel(quantized_latents: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if quantized_latents.ndim != 4:
        raise ValueError(
            f"Expected quantized latents with shape [C, T, H, W], got {tuple(quantized_latents.shape)}."
        )
    scale_view = scales.view(-1, 1, 1, 1).to(dtype=torch.float32)
    return quantized_latents.float() * scale_view


def _encode_quantized_block(latents: torch.Tensor, quant_dtype: str, move_to_cpu: bool = True) -> Dict[str, object]:
    quantized, scales = symmetric_quantize_per_channel(latents, quant_dtype)
    return {
        "quantized": _maybe_to_cpu(quantized, move_to_cpu),
        "scales": _maybe_to_cpu(scales, move_to_cpu),
        "reduced_shape": [int(value) for value in latents.shape],
    }


def _decode_quantized_block(payload: Dict[str, object]) -> torch.Tensor:
    quantized = payload["quantized"]
    if isinstance(quantized, torch.Tensor) and quantized.numel() == 0:
        reduced_shape = tuple(int(value) for value in payload["reduced_shape"])
        return torch.empty(reduced_shape, dtype=torch.float32)
    return symmetric_dequantize_per_channel(
        quantized_latents=payload["quantized"],
        scales=payload["scales"],
    )


def encode_refresh_anchor(
    anchor_latents: torch.Tensor,
    codec_config,
    move_to_cpu: bool = True,
    spatial_factor_override: Optional[int] = None,
) -> Dict[str, object]:
    if anchor_latents.ndim != 4:
        raise ValueError(f"Expected anchor latents with shape [C, T, H, W], got {tuple(anchor_latents.shape)}.")
    if anchor_latents.shape[1] == 0:
        raise ValueError("Anchor latents must contain at least one time step.")

    anchor_spatial_factor = (
        int(spatial_factor_override)
        if spatial_factor_override is not None
        else _as_int(codec_config, "anchor_spatial_factor")
    )
    anchor_quant_dtype = _as_str(codec_config, "anchor_quant_dtype")

    reduced_anchor = anchor_latents
    if anchor_spatial_factor > 1:
        reduced_anchor = trilinear_resize(
            anchor_latents.unsqueeze(0),
            (
                anchor_latents.shape[1],
                max(1, math.ceil(anchor_latents.shape[2] / anchor_spatial_factor)),
                max(1, math.ceil(anchor_latents.shape[3] / anchor_spatial_factor)),
            ),
        ).squeeze(0)
    payload = _encode_quantized_block(reduced_anchor, anchor_quant_dtype, move_to_cpu=move_to_cpu)
    payload["original_shape"] = [int(value) for value in anchor_latents.shape]
    return payload


def decode_refresh_anchor(anchor_payload: Dict[str, object]) -> torch.Tensor:
    original_shape = tuple(int(value) for value in anchor_payload["original_shape"])
    reduced_anchor = _decode_quantized_block(anchor_payload)
    if tuple(reduced_anchor.shape) == original_shape:
        return reduced_anchor.float().contiguous()
    restored_anchor = trilinear_resize(
        reduced_anchor.unsqueeze(0),
        (original_shape[1], original_shape[2], original_shape[3]),
    ).squeeze(0)
    return restored_anchor.float().contiguous()


def encode_tail_p_delta(
    clean_tail_latents: torch.Tensor,
    predicted_tail_latents: torch.Tensor,
    codec_config,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    if clean_tail_latents.shape != predicted_tail_latents.shape:
        raise ValueError(
            "clean_tail_latents and predicted_tail_latents must have the same shape, "
            f"got {tuple(clean_tail_latents.shape)} and {tuple(predicted_tail_latents.shape)}."
        )
    quant_dtype = _as_str(codec_config, "anchor_quant_dtype")
    spatial_factor = _dual_tail_p_delta_spatial_factor(codec_config)
    delta = clean_tail_latents.float() - predicted_tail_latents.float()
    reduced_delta = delta
    if spatial_factor > 1:
        reduced_delta = trilinear_resize(
            delta.unsqueeze(0),
            (
                int(delta.shape[1]),
                max(1, math.ceil(int(delta.shape[2]) / spatial_factor)),
                max(1, math.ceil(int(delta.shape[3]) / spatial_factor)),
            ),
        ).squeeze(0)
    payload = _encode_quantized_block(reduced_delta, quant_dtype, move_to_cpu=move_to_cpu)
    payload["codec_type"] = DUAL_TAIL_P_DELTA_CODEC_TYPE
    payload["original_shape"] = [int(value) for value in clean_tail_latents.shape]
    return payload


def decode_tail_p_delta(tail_payload: Dict[str, object], predicted_tail_latents: torch.Tensor) -> torch.Tensor:
    original_shape = tuple(int(value) for value in tail_payload["original_shape"])
    restored_delta = _decode_quantized_block(tail_payload)
    if tuple(restored_delta.shape) != original_shape:
        restored_delta = trilinear_resize(
            restored_delta.unsqueeze(0),
            (original_shape[1], original_shape[2], original_shape[3]),
        ).squeeze(0)
    return (predicted_tail_latents.float() + restored_delta.float()).contiguous()


def encode_head_p_delta(
    clean_head_latents: torch.Tensor,
    predicted_head_latents: torch.Tensor,
    codec_config,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    if clean_head_latents.shape != predicted_head_latents.shape:
        raise ValueError(
            "clean_head_latents and predicted_head_latents must have the same shape, "
            f"got {tuple(clean_head_latents.shape)} and {tuple(predicted_head_latents.shape)}."
        )
    quant_dtype = _as_str(codec_config, "anchor_quant_dtype")
    spatial_factor = _dual_head_p_delta_spatial_factor(codec_config)
    delta = clean_head_latents.float() - predicted_head_latents.float()
    reduced_delta = delta
    if spatial_factor > 1:
        reduced_delta = trilinear_resize(
            delta.unsqueeze(0),
            (
                int(delta.shape[1]),
                max(1, math.ceil(int(delta.shape[2]) / spatial_factor)),
                max(1, math.ceil(int(delta.shape[3]) / spatial_factor)),
            ),
        ).squeeze(0)
    payload = _encode_quantized_block(reduced_delta, quant_dtype, move_to_cpu=move_to_cpu)
    payload["codec_type"] = DUAL_HEAD_P_DELTA_CODEC_TYPE
    payload["original_shape"] = [int(value) for value in clean_head_latents.shape]
    return payload


def decode_head_p_delta(head_payload: Dict[str, object], predicted_head_latents: torch.Tensor) -> torch.Tensor:
    original_shape = tuple(int(value) for value in head_payload["original_shape"])
    restored_delta = _decode_quantized_block(head_payload)
    if tuple(restored_delta.shape) != original_shape:
        restored_delta = trilinear_resize(
            restored_delta.unsqueeze(0),
            (original_shape[1], original_shape[2], original_shape[3]),
        ).squeeze(0)
    return (predicted_head_latents.float() + restored_delta.float()).contiguous()


def encode_head_soft_pool(
    clean_head_latents: torch.Tensor,
    codec_config,
    move_to_cpu: bool = True,
    spatial_factor_override: Optional[int] = None,
) -> Dict[str, object]:
    if clean_head_latents.ndim != 4:
        raise ValueError(
            f"Expected clean_head_latents with shape [C, T, H, W], got {tuple(clean_head_latents.shape)}."
        )
    quant_dtype = _as_str(codec_config, "anchor_quant_dtype")
    spatial_factor = (
        int(spatial_factor_override)
        if spatial_factor_override is not None
        else _dual_head_soft_pool_spatial_factor(codec_config)
    )
    reduced_head = clean_head_latents.float()
    if spatial_factor > 1:
        reduced_head = F.adaptive_avg_pool3d(
            clean_head_latents.unsqueeze(0).float(),
            output_size=(
                int(clean_head_latents.shape[1]),
                max(1, math.ceil(int(clean_head_latents.shape[2]) / spatial_factor)),
                max(1, math.ceil(int(clean_head_latents.shape[3]) / spatial_factor)),
            ),
        ).squeeze(0)
    payload = _encode_quantized_block(reduced_head, quant_dtype, move_to_cpu=move_to_cpu)
    payload["codec_type"] = DUAL_HEAD_SOFT_POOL_CODEC_TYPE
    payload["original_shape"] = [int(value) for value in clean_head_latents.shape]
    return payload


def decode_head_soft_pool(head_payload: Dict[str, object]) -> torch.Tensor:
    original_shape = tuple(int(value) for value in head_payload["original_shape"])
    restored_head = _decode_quantized_block(head_payload)
    if tuple(restored_head.shape) != original_shape:
        restored_head = trilinear_resize(
            restored_head.unsqueeze(0),
            (original_shape[1], original_shape[2], original_shape[3]),
        ).squeeze(0)
    return restored_head.float().contiguous()


def encode_global_keyframe(
    global_keyframe_latents: torch.Tensor,
    codec_config,
    move_to_cpu: bool = True,
) -> Tuple[Dict[str, object], torch.Tensor]:
    if global_keyframe_latents.ndim != 4:
        raise ValueError(
            "Expected global keyframe latents with shape [C, T, H, W], "
            f"got {tuple(global_keyframe_latents.shape)}."
        )
    if int(global_keyframe_latents.shape[1]) != 1:
        raise ValueError(
            "Global keyframe must contain exactly one time step, "
            f"got shape {tuple(global_keyframe_latents.shape)}."
        )

    codec_type = _global_keyframe_codec_type(codec_config)
    if codec_type == RAW_GLOBAL_KEYFRAME_CODEC_TYPE:
        keyframe_dtype = torch.float16 if _as_str(codec_config, "keyframe_dtype") == "float16" else torch.float32
        stored_keyframe = _maybe_to_cpu(global_keyframe_latents.to(keyframe_dtype).contiguous(), move_to_cpu)
        decoded_keyframe = global_keyframe_latents.float().contiguous()
        return {
            "codec_type": codec_type,
            "global_keyframe": stored_keyframe,
        }, decoded_keyframe

    if codec_type == INT8_REFRESH_LIKE_GLOBAL_KEYFRAME_CODEC_TYPE:
        quant_dtype = _global_keyframe_quant_dtype(codec_config)
        spatial_factor = _global_keyframe_spatial_factor(codec_config)
        reduced_keyframe = global_keyframe_latents.float()
        if spatial_factor > 1:
            reduced_keyframe = trilinear_resize(
                global_keyframe_latents.unsqueeze(0).float(),
                (
                    int(global_keyframe_latents.shape[1]),
                    max(1, math.ceil(int(global_keyframe_latents.shape[2]) / spatial_factor)),
                    max(1, math.ceil(int(global_keyframe_latents.shape[3]) / spatial_factor)),
                ),
            ).squeeze(0)
        keyframe_payload = _encode_quantized_block(reduced_keyframe, quant_dtype, move_to_cpu=move_to_cpu)
        keyframe_payload["original_shape"] = [int(value) for value in global_keyframe_latents.shape]
        decoded_keyframe = decode_refresh_anchor(keyframe_payload).to(
            device=global_keyframe_latents.device,
            dtype=torch.float32,
        )
        return {
            "codec_type": codec_type,
            "global_keyframe_payload": keyframe_payload,
        }, decoded_keyframe.contiguous()

    raise ValueError(f"Unsupported global keyframe codec type: {codec_type}.")


def decode_global_keyframe(codec_payload: Dict[str, object]) -> torch.Tensor:
    if codec_payload.get("global_keyframe_payload") is not None:
        return decode_refresh_anchor(codec_payload["global_keyframe_payload"]).float().contiguous()
    global_keyframe = codec_payload.get("global_keyframe")
    if isinstance(global_keyframe, torch.Tensor):
        return global_keyframe.float().contiguous()
    raise KeyError("Missing global keyframe payload. Expected `global_keyframe` or `global_keyframe_payload`.")


def _section_anchor_span(section_length: int, codec_config) -> int:
    return min(_as_int(codec_config, "anchor_span_latents"), section_length)


def _build_section_error_weights(section_length: int, device: torch.device) -> torch.Tensor:
    if section_length <= 0:
        raise ValueError(f"section_length must be > 0, got {section_length}.")
    if section_length == 1:
        return torch.ones(1, 1, 1, 1, device=device, dtype=torch.float32)
    weights = torch.linspace(1.0, 2.0, steps=section_length, device=device, dtype=torch.float32)
    weights[0] += 1.0
    weights[-1] += 1.0
    return weights.view(1, section_length, 1, 1)


def _predict_section_from_history(
    history_latents: torch.Tensor,
    section_shape: Tuple[int, int, int, int],
) -> torch.Tensor:
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
        predicted = trilinear_resize(predicted.unsqueeze(0), (section_length, height, width)).squeeze(0)
    return predicted.float().contiguous()


def _overlay_anchor_blocks(
    section_prediction: torch.Tensor,
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
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


def _interpolate_between_anchor_blocks(
    section_prediction: torch.Tensor,
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
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


def _build_proxy_section_from_history(
    history_latents: torch.Tensor,
    section_shape: Tuple[int, int, int, int],
    anchor_blocks: Sequence[Tuple[int, torch.Tensor]],
) -> torch.Tensor:
    prediction = _predict_section_from_history(history_latents, section_shape)
    prediction = _overlay_anchor_blocks(prediction, anchor_blocks)
    prediction = _interpolate_between_anchor_blocks(prediction, anchor_blocks)
    prediction = _overlay_anchor_blocks(prediction, anchor_blocks)
    return prediction.contiguous()


def _compute_proxy_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(prediction.shape)} does not match target shape {tuple(target.shape)}."
        )
    weights = _build_section_error_weights(prediction.shape[1], prediction.device)
    sq_error = (prediction.float() - target.float()).pow(2)
    return float((sq_error * weights).mean().item())


def _compute_boundary_jump_l1(history_latents: torch.Tensor, section_latents: torch.Tensor) -> float:
    if history_latents.shape[1] == 0 or section_latents.shape[1] == 0:
        return 0.0
    return float((section_latents[:, :1].float() - history_latents[:, -1:].float()).abs().mean().item())


def _estimate_refresh_anchor_bytes(anchor_payload: Dict[str, object]) -> int:
    total_bytes = _estimate_payload_data_bytes(anchor_payload)
    total_bytes += _estimate_payload_shape_metadata_bytes(anchor_payload)
    return int(total_bytes)


def _estimate_payload_data_bytes(payload: Dict[str, object]) -> int:
    total_bytes = 0
    for tensor_key in ("quantized", "scales"):
        value = payload.get(tensor_key)
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()
    if "ste_bottleneck" in payload:
        reduced_shape = [int(value) for value in payload.get("reduced_shape", [])]
        if reduced_shape:
            channel_count = int(reduced_shape[0])
            bottleneck_elements = 1
            for value in reduced_shape:
                bottleneck_elements *= int(value)
            total_bytes += bottleneck_elements
            total_bytes += channel_count * 4
    return int(total_bytes)


def _estimate_payload_shape_metadata_bytes(payload: Dict[str, object]) -> int:
    total_bytes = 0
    for shape_key in ("reduced_shape", "original_shape"):
        total_bytes += len(payload.get(shape_key, [])) * 4
    return int(total_bytes)


def _estimate_global_keyframe_bytes_from_encoded_entry(encoded_global_keyframe: Dict[str, object]) -> int:
    stored_keyframe = encoded_global_keyframe.get("global_keyframe")
    if isinstance(stored_keyframe, torch.Tensor):
        return int(stored_keyframe.numel() * stored_keyframe.element_size())
    keyframe_payload = encoded_global_keyframe.get("global_keyframe_payload")
    if isinstance(keyframe_payload, dict):
        return _estimate_refresh_anchor_bytes(keyframe_payload)
    return 0


def _estimate_global_keyframe_bytes(codec_payload: Dict[str, object]) -> int:
    global_keyframe = codec_payload.get("global_keyframe")
    if isinstance(global_keyframe, torch.Tensor):
        return int(global_keyframe.numel() * global_keyframe.element_size())
    global_keyframe_payload = codec_payload.get("global_keyframe_payload")
    if isinstance(global_keyframe_payload, dict):
        return _estimate_refresh_anchor_bytes(global_keyframe_payload)
    return 0


def _empty_low_codec_byte_breakdown() -> Dict[str, int]:
    return {key: 0 for key in LOW_CODEC_BYTE_BREAKDOWN_KEYS}


def estimate_anchor_plus_tail_codec_byte_breakdown(codec_payload: Dict[str, object]) -> Dict[str, int]:
    breakdown = _empty_low_codec_byte_breakdown()

    global_keyframe = codec_payload.get("global_keyframe")
    if isinstance(global_keyframe, torch.Tensor):
        breakdown["global_keyframe_bytes"] += int(global_keyframe.numel() * global_keyframe.element_size())
    global_keyframe_payload = codec_payload.get("global_keyframe_payload")
    if isinstance(global_keyframe_payload, dict):
        breakdown["global_keyframe_bytes"] += _estimate_payload_data_bytes(global_keyframe_payload)
        breakdown["metadata_bytes"] += _estimate_payload_shape_metadata_bytes(global_keyframe_payload)

    if codec_payload.get("section_payloads"):
        metadata_values: List[int] = [len(codec_payload.get("chunk_lengths", []))]
        metadata_values.extend(int(value) for value in codec_payload.get("chunk_lengths", []))
        for start, end in codec_payload.get("section_ranges", []):
            metadata_values.extend([int(start), int(end)])
        breakdown["metadata_bytes"] += len(metadata_values) * 4

        for section_payload in codec_payload.get("section_payloads", []):
            breakdown["metadata_bytes"] += 4  # mode id
            heuristic_scores = section_payload.get("heuristic_scores", {})
            breakdown["metadata_bytes"] += len(heuristic_scores) * 4

            mode = str(section_payload.get("mode", SINGLE_REFRESH_SECTION_MODE))
            for anchor_index, anchor_block in enumerate(section_payload.get("anchor_blocks", [])):
                breakdown["metadata_bytes"] += 8  # start + length
                anchor_payload = anchor_block.get("payload")
                if not isinstance(anchor_payload, dict):
                    continue
                payload_data_bytes = _estimate_payload_data_bytes(anchor_payload)
                breakdown["metadata_bytes"] += _estimate_payload_shape_metadata_bytes(anchor_payload)
                if mode == DUAL_REFRESH_SECTION_MODE:
                    target_key = "dual_head_bytes" if anchor_index == 0 else "dual_tail_presidual_bytes"
                else:
                    target_key = "single_head_bytes"
                breakdown[target_key] += payload_data_bytes
        return {key: int(value) for key, value in breakdown.items()}

    for payload in codec_payload.get("section_anchor_payloads", []):
        breakdown["single_head_bytes"] += _estimate_payload_data_bytes(payload)
        breakdown["metadata_bytes"] += _estimate_payload_shape_metadata_bytes(payload)
    for payload in codec_payload.get("section_tail_payloads", []):
        breakdown["dual_tail_presidual_bytes"] += _estimate_payload_data_bytes(payload)
        breakdown["metadata_bytes"] += _estimate_payload_shape_metadata_bytes(payload)

    metadata_values = [len(codec_payload.get("chunk_lengths", []))]
    metadata_values.extend(int(value) for value in codec_payload.get("chunk_lengths", []))
    for start, end in codec_payload.get("section_ranges", []):
        metadata_values.extend([int(start), int(end)])
    breakdown["metadata_bytes"] += len(metadata_values) * 4
    return {key: int(value) for key, value in breakdown.items()}


def _analyze_section_mode_candidates(
    section_latents: torch.Tensor,
    history_latents: torch.Tensor,
    codec_config,
    sections_since_refresh: int,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    section_length = int(section_latents.shape[1])
    anchor_span = _section_anchor_span(section_length, codec_config)
    head_clean_block = (0, section_latents[:, :anchor_span].contiguous())
    tail_start = max(0, section_length - anchor_span)
    tail_clean_block = (tail_start, section_latents[:, tail_start:].contiguous())

    full_head_payload = encode_refresh_anchor(head_clean_block[1], codec_config, move_to_cpu=move_to_cpu)
    full_head_decoded_block = (
        int(head_clean_block[0]),
        decode_refresh_anchor(full_head_payload).to(device=section_latents.device, dtype=torch.float32),
    )
    full_head_encoded_block = {
        "start": int(head_clean_block[0]),
        "length": int(head_clean_block[1].shape[1]),
        "payload": full_head_payload,
        "bytes": _estimate_refresh_anchor_bytes(full_head_payload),
    }

    single_head_codec_type = _single_head_codec_type(codec_config)
    if single_head_codec_type == SINGLE_HEAD_SOFT_POOL_CODEC_TYPE:
        single_head_payload = encode_head_soft_pool(
            clean_head_latents=head_clean_block[1],
            codec_config=codec_config,
            move_to_cpu=move_to_cpu,
            spatial_factor_override=_single_head_soft_pool_spatial_factor(codec_config),
        )
        single_decoded_blocks = [
            (
                int(head_clean_block[0]),
                decode_head_soft_pool(single_head_payload).to(
                    device=section_latents.device,
                    dtype=torch.float32,
                ),
            )
        ]
        single_encoded_blocks = [
            {
                "start": int(head_clean_block[0]),
                "length": int(head_clean_block[1].shape[1]),
                "payload": single_head_payload,
                "bytes": _estimate_refresh_anchor_bytes(single_head_payload),
                "head_codec_type": SINGLE_HEAD_SOFT_POOL_CODEC_TYPE,
            }
        ]
    else:
        single_decoded_blocks = [full_head_decoded_block]
        single_encoded_blocks = [dict(full_head_encoded_block)]

    none_prediction = _build_proxy_section_from_history(
        history_latents,
        tuple(int(value) for value in section_latents.shape),
        [],
    )

    dual_decoded_blocks = [full_head_decoded_block]
    dual_encoded_blocks = [dict(full_head_encoded_block)]
    dual_head_codec_type = _dual_head_codec_type(codec_config)
    dual_head_spatial_factor = _dual_head_anchor_spatial_factor(codec_config)
    if dual_head_codec_type == DUAL_HEAD_P_DELTA_CODEC_TYPE:
        head_prediction = none_prediction[:, :anchor_span].contiguous()
        dual_head_payload = encode_head_p_delta(
            clean_head_latents=head_clean_block[1],
            predicted_head_latents=head_prediction,
            codec_config=codec_config,
            move_to_cpu=move_to_cpu,
        )
        dual_decoded_blocks[0] = (
            int(head_clean_block[0]),
            decode_head_p_delta(dual_head_payload, head_prediction).to(device=section_latents.device, dtype=torch.float32),
        )
        dual_encoded_blocks[0] = {
            "start": int(head_clean_block[0]),
            "length": int(head_clean_block[1].shape[1]),
            "payload": dual_head_payload,
            "bytes": _estimate_refresh_anchor_bytes(dual_head_payload),
            "head_codec_type": DUAL_HEAD_P_DELTA_CODEC_TYPE,
        }
    elif dual_head_codec_type == DUAL_HEAD_SOFT_POOL_CODEC_TYPE:
        dual_head_payload = encode_head_soft_pool(
            clean_head_latents=head_clean_block[1],
            codec_config=codec_config,
            move_to_cpu=move_to_cpu,
        )
        dual_decoded_blocks[0] = (
            int(head_clean_block[0]),
            decode_head_soft_pool(dual_head_payload).to(device=section_latents.device, dtype=torch.float32),
        )
        dual_encoded_blocks[0] = {
            "start": int(head_clean_block[0]),
            "length": int(head_clean_block[1].shape[1]),
            "payload": dual_head_payload,
            "bytes": _estimate_refresh_anchor_bytes(dual_head_payload),
            "head_codec_type": DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
        }
    elif dual_head_spatial_factor is not None and dual_head_spatial_factor != _as_int(codec_config, "anchor_spatial_factor"):
        dual_head_payload = encode_refresh_anchor(
            head_clean_block[1],
            codec_config,
            move_to_cpu=move_to_cpu,
            spatial_factor_override=dual_head_spatial_factor,
        )
        dual_decoded_blocks[0] = (
            int(head_clean_block[0]),
            decode_refresh_anchor(dual_head_payload).to(device=section_latents.device, dtype=torch.float32),
        )
        dual_encoded_blocks[0] = {
            "start": int(head_clean_block[0]),
            "length": int(head_clean_block[1].shape[1]),
            "payload": dual_head_payload,
            "bytes": _estimate_refresh_anchor_bytes(dual_head_payload),
        }
    single_prediction = _build_proxy_section_from_history(
        history_latents,
        tuple(int(value) for value in section_latents.shape),
        single_decoded_blocks,
    )
    if tail_start >= anchor_span and tail_start < section_length:
        head_conditioned_prediction = _build_proxy_section_from_history(
            history_latents,
            tuple(int(value) for value in section_latents.shape),
            dual_decoded_blocks[:1],
        )
        tail_prediction = head_conditioned_prediction[
            :, tail_start : tail_start + int(tail_clean_block[1].shape[1])
        ].contiguous()
        if _dual_tail_codec_type(codec_config) == DUAL_TAIL_P_DELTA_CODEC_TYPE:
            tail_payload = encode_tail_p_delta(
                clean_tail_latents=tail_clean_block[1],
                predicted_tail_latents=tail_prediction,
                codec_config=codec_config,
                move_to_cpu=move_to_cpu,
            )
            decoded_tail = decode_tail_p_delta(tail_payload, tail_prediction).to(
                device=section_latents.device,
                dtype=torch.float32,
            )
            tail_codec_type = DUAL_TAIL_P_DELTA_CODEC_TYPE
        else:
            tail_payload = encode_refresh_anchor(
                tail_clean_block[1],
                codec_config,
                move_to_cpu=move_to_cpu,
                spatial_factor_override=_dual_tail_anchor_spatial_factor(codec_config),
            )
            decoded_tail = decode_refresh_anchor(tail_payload).to(
                device=section_latents.device,
                dtype=torch.float32,
            )
            tail_codec_type = DUAL_TAIL_ANCHOR_CODEC_TYPE
        dual_decoded_blocks.append(
            (
                int(tail_clean_block[0]),
                decoded_tail,
            )
        )
        dual_encoded_blocks.append(
            {
                "start": int(tail_clean_block[0]),
                "length": int(tail_clean_block[1].shape[1]),
                "payload": tail_payload,
                "bytes": _estimate_refresh_anchor_bytes(tail_payload),
                "tail_codec_type": tail_codec_type,
            }
        )

    dual_prediction = _build_proxy_section_from_history(
        history_latents,
        tuple(int(value) for value in section_latents.shape),
        dual_decoded_blocks,
    )

    error_none = _compute_proxy_error(none_prediction, section_latents)
    error_single = _compute_proxy_error(single_prediction, section_latents)
    error_dual = _compute_proxy_error(dual_prediction, section_latents)
    gain_single = max(0.0, error_none - error_single)
    gain_dual = max(0.0, error_single - error_dual)
    gain_predict_to_dual = max(0.0, error_none - error_dual)
    gain_ratio_single = gain_single / max(error_none, EPS)
    gain_ratio_dual = gain_dual / max(error_single, EPS)
    boundary_jump_l1 = _compute_boundary_jump_l1(history_latents, section_latents)

    max_gap = max(
        0,
        int(
            _get_optional_config_value(
                codec_config,
                "max_predict_only_gap_sections",
                DEFAULT_MAX_PREDICT_ONLY_GAP_SECTIONS,
            )
        ),
    )
    single_gain_threshold = _as_float(
        codec_config,
        "single_refresh_gain_threshold",
        DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD,
    )
    dual_gain_threshold = _as_float(
        codec_config,
        "dual_refresh_gain_threshold",
        DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD,
    )
    boundary_jump_threshold = _as_float(
        codec_config,
        "boundary_jump_threshold",
        DEFAULT_BOUNDARY_JUMP_THRESHOLD,
    )
    cut_detection_threshold = _as_float(
        codec_config,
        "cut_detection_threshold",
        DEFAULT_CUT_DETECTION_THRESHOLD,
    )

    force_single = sections_since_refresh >= max_gap or boundary_jump_l1 >= boundary_jump_threshold
    force_dual = boundary_jump_l1 >= cut_detection_threshold and len(dual_decoded_blocks) > 1

    minimum_mode = PREDICT_ONLY_SECTION_MODE
    if force_dual:
        minimum_mode = DUAL_REFRESH_SECTION_MODE
    elif force_single:
        minimum_mode = SINGLE_REFRESH_SECTION_MODE

    return {
        "minimum_mode": minimum_mode,
        "clean_blocks_by_mode": {
            PREDICT_ONLY_SECTION_MODE: [],
            SINGLE_REFRESH_SECTION_MODE: [(int(head_clean_block[0]), head_clean_block[1].contiguous())],
            DUAL_REFRESH_SECTION_MODE: [
                (int(block_start), block_latents.contiguous())
                for block_start, block_latents in (
                    [(int(head_clean_block[0]), head_clean_block[1])]
                    + (
                        [(int(tail_clean_block[0]), tail_clean_block[1])]
                        if len(dual_decoded_blocks) > 1
                        else []
                    )
                )
            ],
        },
        "decoded_blocks_by_mode": {
            PREDICT_ONLY_SECTION_MODE: [],
            SINGLE_REFRESH_SECTION_MODE: list(single_decoded_blocks),
            DUAL_REFRESH_SECTION_MODE: list(dual_decoded_blocks),
        },
        "encoded_blocks_by_mode": {
            PREDICT_ONLY_SECTION_MODE: [],
            SINGLE_REFRESH_SECTION_MODE: list(single_encoded_blocks),
            DUAL_REFRESH_SECTION_MODE: list(dual_encoded_blocks),
        },
        "full_head_encoded_block": dict(full_head_encoded_block),
        "single_head_codec_type": single_head_codec_type,
        "mode_total_anchor_bytes": {
            PREDICT_ONLY_SECTION_MODE: 0,
            SINGLE_REFRESH_SECTION_MODE: sum(
                int(anchor_block["bytes"]) + 8 for anchor_block in single_encoded_blocks
            ),
            DUAL_REFRESH_SECTION_MODE: sum(
                int(anchor_block["bytes"]) + 8 for anchor_block in dual_encoded_blocks
            ),
        },
        "heuristic_scores": {
            "error_none": float(error_none),
            "error_single": float(error_single),
            "error_dual": float(error_dual),
            "gain_single": float(gain_single),
            "gain_dual": float(gain_dual),
            "gain_predict_to_dual": float(gain_predict_to_dual),
            "gain_ratio_single": float(gain_ratio_single),
            "gain_ratio_dual": float(gain_ratio_dual),
            "boundary_jump_l1": float(boundary_jump_l1),
        },
    }


def _select_section_mode_from_analysis(
    section_analysis: Dict[str, object],
    codec_config,
) -> str:
    heuristic_scores = section_analysis["heuristic_scores"]
    gain_ratio_single = float(heuristic_scores["gain_ratio_single"])
    gain_ratio_dual = float(heuristic_scores["gain_ratio_dual"])
    minimum_mode = str(section_analysis["minimum_mode"])

    single_gain_threshold = _as_float(
        codec_config,
        "single_refresh_gain_threshold",
        DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD,
    )
    dual_gain_threshold = _as_float(
        codec_config,
        "dual_refresh_gain_threshold",
        DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD,
    )

    if minimum_mode == DUAL_REFRESH_SECTION_MODE:
        return DUAL_REFRESH_SECTION_MODE
    if minimum_mode == SINGLE_REFRESH_SECTION_MODE:
        if gain_ratio_dual >= dual_gain_threshold and len(section_analysis["decoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 1:
            return DUAL_REFRESH_SECTION_MODE
        return SINGLE_REFRESH_SECTION_MODE
    if gain_ratio_single < single_gain_threshold:
        return PREDICT_ONLY_SECTION_MODE
    if gain_ratio_dual >= dual_gain_threshold and len(section_analysis["decoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 1:
        return DUAL_REFRESH_SECTION_MODE
    return SINGLE_REFRESH_SECTION_MODE


def _select_section_mode(
    section_latents: torch.Tensor,
    history_latents: torch.Tensor,
    codec_config,
    sections_since_refresh: int,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    section_analysis = _analyze_section_mode_candidates(
        section_latents=section_latents,
        history_latents=history_latents,
        codec_config=codec_config,
        sections_since_refresh=sections_since_refresh,
        move_to_cpu=move_to_cpu,
    )
    mode = _select_section_mode_from_analysis(section_analysis, codec_config)
    return {
        "mode": mode,
        "anchor_blocks": list(section_analysis["decoded_blocks_by_mode"][mode]),
        "encoded_anchor_blocks": list(section_analysis["encoded_blocks_by_mode"][mode]),
        "heuristic_scores": dict(section_analysis["heuristic_scores"]),
    }


def _select_adaptive_dual_head_full_indices(
    section_analyses: Sequence[Dict[str, object]],
    selected_modes: Sequence[str],
    codec_config,
) -> set[int]:
    if _dual_head_codec_type(codec_config) != DUAL_HEAD_ANCHOR_CODEC_TYPE:
        return set()
    dual_head_spatial_factor = _dual_head_anchor_spatial_factor(codec_config)
    if dual_head_spatial_factor is None:
        return set()
    if dual_head_spatial_factor == _as_int(codec_config, "anchor_spatial_factor"):
        return set()

    full_ratio = _adaptive_dual_head_full_ratio(codec_config)
    if full_ratio <= 0.0:
        return set()

    scored_dual_sections: List[Tuple[Tuple[float, float, float], int]] = []
    for section_index, (section_analysis, selected_mode) in enumerate(zip(section_analyses, selected_modes)):
        if str(selected_mode) != DUAL_REFRESH_SECTION_MODE:
            continue
        dual_blocks = section_analysis["encoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]
        if not dual_blocks:
            continue
        heuristic_scores = section_analysis["heuristic_scores"]
        hard_score = (
            float(heuristic_scores["boundary_jump_l1"]),
            float(heuristic_scores["gain_ratio_dual"]),
            float(heuristic_scores["error_single"]),
        )
        scored_dual_sections.append((hard_score, section_index))

    if not scored_dual_sections:
        return set()

    full_count = min(len(scored_dual_sections), int(math.ceil(full_ratio * len(scored_dual_sections))))
    if full_count <= 0:
        return set()

    scored_dual_sections.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return {section_index for _, section_index in scored_dual_sections[:full_count]}


def _select_adaptive_dual_head_soft_pool_hard_indices(
    section_analyses: Sequence[Dict[str, object]],
    selected_modes: Sequence[str],
    codec_config,
) -> set[int]:
    if _dual_head_codec_type(codec_config) != DUAL_HEAD_SOFT_POOL_CODEC_TYPE:
        return set()
    hard_factor = _adaptive_dual_head_soft_pool_hard_factor(codec_config)
    if hard_factor is None:
        return set()
    if hard_factor == _dual_head_soft_pool_spatial_factor(codec_config):
        return set()

    hard_ratio = _adaptive_dual_head_soft_pool_hard_ratio(codec_config)
    if hard_ratio <= 0.0:
        return set()

    scored_dual_sections: List[Tuple[Tuple[float, float, float], int]] = []
    for section_index, (section_analysis, selected_mode) in enumerate(zip(section_analyses, selected_modes)):
        if str(selected_mode) != DUAL_REFRESH_SECTION_MODE:
            continue
        dual_blocks = section_analysis["encoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]
        if not dual_blocks:
            continue
        heuristic_scores = section_analysis["heuristic_scores"]
        hard_score = (
            float(heuristic_scores["boundary_jump_l1"]),
            float(heuristic_scores["gain_ratio_dual"]),
            float(heuristic_scores["error_single"]),
        )
        scored_dual_sections.append((hard_score, section_index))

    if not scored_dual_sections:
        return set()

    hard_count = min(len(scored_dual_sections), int(math.ceil(hard_ratio * len(scored_dual_sections))))
    if hard_count <= 0:
        return set()

    scored_dual_sections.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return {section_index for _, section_index in scored_dual_sections[:hard_count]}


def _estimate_budget_fixed_bytes(
    global_keyframe_bytes: int,
    chunk_lengths: Sequence[int],
    section_ranges: Sequence[Tuple[int, int]],
    section_analyses: Sequence[Dict[str, object]],
) -> int:
    total_bytes = 0
    total_bytes += int(global_keyframe_bytes)
    metadata_values: List[int] = [len(chunk_lengths)]
    metadata_values.extend(int(value) for value in chunk_lengths)
    for start, end in section_ranges:
        metadata_values.extend([int(start), int(end)])
    total_bytes += len(metadata_values) * 4
    for section_analysis in section_analyses:
        total_bytes += 4  # mode id
        total_bytes += len(section_analysis["heuristic_scores"]) * 4
    return int(total_bytes)


def _resolve_budget_target_bytes(total_pixels: Optional[int], target_low_bpp: Optional[float]) -> Optional[int]:
    if total_pixels is None or target_low_bpp is None:
        return None
    if total_pixels <= 0:
        raise ValueError(f"total_pixels must be > 0 when target_low_bpp is set, got {total_pixels}.")
    return max(0, int(math.floor(float(target_low_bpp) * float(total_pixels) / 8.0)))


def _allocate_section_modes_with_budget(
    section_analyses: Sequence[Dict[str, object]],
    codec_config,
    global_keyframe_bytes: int,
    chunk_lengths: Sequence[int],
    section_ranges: Sequence[Tuple[int, int]],
    total_pixels: int,
) -> Dict[str, object]:
    target_low_bpp = _as_optional_float(codec_config, "target_low_bpp")
    target_low_bytes = _resolve_budget_target_bytes(total_pixels, target_low_bpp)
    if target_low_bytes is None:
        raise ValueError("Budget allocation requires target_low_bpp and total_pixels.")

    selected_modes = [str(section_analysis["minimum_mode"]) for section_analysis in section_analyses]
    fixed_bytes = _estimate_budget_fixed_bytes(global_keyframe_bytes, chunk_lengths, section_ranges, section_analyses)
    current_bytes = fixed_bytes + sum(
        int(section_analysis["mode_total_anchor_bytes"][mode])
        for section_analysis, mode in zip(section_analyses, selected_modes)
    )

    while current_bytes < target_low_bytes:
        best_candidate = None
        for section_index, section_analysis in enumerate(section_analyses):
            current_mode = selected_modes[section_index]
            heuristic_scores = section_analysis["heuristic_scores"]
            mode_total_anchor_bytes = section_analysis["mode_total_anchor_bytes"]

            candidates: List[Tuple[str, float, int]] = []
            if current_mode == PREDICT_ONLY_SECTION_MODE:
                single_extra_bytes = int(mode_total_anchor_bytes[SINGLE_REFRESH_SECTION_MODE])
                if single_extra_bytes > 0:
                    candidates.append(
                        (
                            SINGLE_REFRESH_SECTION_MODE,
                            float(heuristic_scores["gain_single"]),
                            single_extra_bytes,
                        )
                    )
                if len(section_analysis["decoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 1:
                    dual_extra_bytes = int(mode_total_anchor_bytes[DUAL_REFRESH_SECTION_MODE])
                    if dual_extra_bytes > 0:
                        candidates.append(
                            (
                                DUAL_REFRESH_SECTION_MODE,
                                float(heuristic_scores["gain_predict_to_dual"]),
                                dual_extra_bytes,
                            )
                        )
            elif current_mode == SINGLE_REFRESH_SECTION_MODE:
                if len(section_analysis["decoded_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 1:
                    dual_extra_bytes = int(
                        mode_total_anchor_bytes[DUAL_REFRESH_SECTION_MODE]
                        - mode_total_anchor_bytes[SINGLE_REFRESH_SECTION_MODE]
                    )
                    if dual_extra_bytes > 0:
                        candidates.append(
                            (
                                DUAL_REFRESH_SECTION_MODE,
                                float(heuristic_scores["gain_dual"]),
                                dual_extra_bytes,
                            )
                        )

            for next_mode, gain_value, extra_bytes in candidates:
                if gain_value <= 0.0:
                    continue
                if current_bytes + extra_bytes > target_low_bytes:
                    continue
                utility = gain_value / max(float(extra_bytes), 1.0)
                if best_candidate is None:
                    best_candidate = (utility, gain_value, -extra_bytes, section_index, next_mode, extra_bytes)
                    continue
                if (utility, gain_value, -extra_bytes) > best_candidate[:3]:
                    best_candidate = (utility, gain_value, -extra_bytes, section_index, next_mode, extra_bytes)

        if best_candidate is None:
            break
        _utility, _gain_value, _neg_extra_bytes, section_index, next_mode, extra_bytes = best_candidate
        selected_modes[section_index] = next_mode
        current_bytes += int(extra_bytes)

    return {
        "selected_modes": selected_modes,
        "target_low_bpp": target_low_bpp,
        "target_low_bytes": target_low_bytes,
        "estimated_selected_bytes": int(current_bytes),
        "estimated_fixed_bytes": int(fixed_bytes),
    }

    if not force_single and gain_ratio_single < single_gain_threshold:
        mode = PREDICT_ONLY_SECTION_MODE
        chosen_blocks: List[Tuple[int, torch.Tensor]] = []
    elif prefer_dual or (len(dual_blocks) > 1 and gain_ratio_dual >= dual_gain_threshold):
        mode = DUAL_REFRESH_SECTION_MODE
        chosen_blocks = dual_blocks
    else:
        mode = SINGLE_REFRESH_SECTION_MODE
        chosen_blocks = [head_block]

    return {
        "mode": mode,
        "anchor_blocks": chosen_blocks,
        "heuristic_scores": {
            "error_none": float(error_none),
            "error_single": float(error_single),
            "error_dual": float(error_dual),
            "gain_single": float(gain_single),
            "gain_dual": float(gain_dual),
            "gain_ratio_single": float(gain_ratio_single),
            "gain_ratio_dual": float(gain_ratio_dual),
            "boundary_jump_l1": float(boundary_jump_l1),
        },
    }


def encode_section_tail_residual(
    section_tail_latents: torch.Tensor,
    decoded_anchor_latents: torch.Tensor,
    codec_config,
    learned_tail_codec=None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    if section_tail_latents.ndim != 4:
        raise ValueError(
            f"Expected section tail latents with shape [C, T, H, W], got {tuple(section_tail_latents.shape)}."
        )
    original_shape = [int(value) for value in section_tail_latents.shape]
    if section_tail_latents.shape[1] == 0:
        empty_quantized = torch.empty(0, dtype=torch.int8, device=section_tail_latents.device)
        empty_scales = torch.empty(0, dtype=torch.float32, device=section_tail_latents.device)
        return {
            "tail_codec_type": _tail_codec_type(codec_config),
            "quantized": _maybe_to_cpu(empty_quantized, move_to_cpu),
            "scales": _maybe_to_cpu(empty_scales, move_to_cpu),
            "reduced_shape": [
                int(section_tail_latents.shape[0]),
                0,
                int(section_tail_latents.shape[2]),
                int(section_tail_latents.shape[3]),
            ],
            "original_shape": original_shape,
        }

    anchor_reference = (
        decoded_anchor_latents[:, -1:]
        .to(device=section_tail_latents.device, dtype=torch.float32)
        .expand(-1, section_tail_latents.shape[1], -1, -1)
    )
    residual_tail = section_tail_latents.float() - anchor_reference
    tail_codec_type = _tail_codec_type(codec_config)
    if tail_codec_type == LEARNED_TAIL_CODEC_TYPE:
        if learned_tail_codec is None:
            raise ValueError("learned_tail_codec must be provided when tail_codec_type=learned_cnn_v1.")
        if use_ste_quant:
            payload = learned_tail_codec.encode_to_ste_payload(residual_tail.unsqueeze(0))
        else:
            payload = learned_tail_codec.encode_to_quantized_payload(residual_tail.unsqueeze(0))
        payload["tail_codec_type"] = LEARNED_TAIL_CODEC_TYPE
        payload["original_shape"] = original_shape
        return payload

    reduced_tail = trilinear_resize(
        residual_tail.unsqueeze(0),
        (
            max(1, math.ceil(section_tail_latents.shape[1] / _as_int(codec_config, "temporal_factor"))),
            max(1, math.ceil(section_tail_latents.shape[2] / _as_int(codec_config, "spatial_factor"))),
            max(1, math.ceil(section_tail_latents.shape[3] / _as_int(codec_config, "spatial_factor"))),
        ),
    ).squeeze(0)
    payload = _encode_quantized_block(
        reduced_tail,
        _as_str(codec_config, "quant_dtype"),
        move_to_cpu=move_to_cpu,
    )
    payload["tail_codec_type"] = TRILINEAR_TAIL_CODEC_TYPE
    payload["original_shape"] = original_shape
    return payload


def decode_section_tail_residual(
    tail_payload: Dict[str, object],
    decoded_anchor_latents: torch.Tensor,
    learned_tail_codec=None,
) -> torch.Tensor:
    original_shape = tuple(int(value) for value in tail_payload["original_shape"])
    if original_shape[1] == 0:
        return torch.empty(
            original_shape,
            dtype=torch.float32,
            device=decoded_anchor_latents.device,
        )

    tail_codec_type = str(tail_payload.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    if tail_codec_type == LEARNED_TAIL_CODEC_TYPE:
        if learned_tail_codec is None:
            raise ValueError("learned_tail_codec must be provided to decode learned tail payloads.")
        restored_residual = learned_tail_codec.decode_payload(tail_payload)
        anchor_source = decoded_anchor_latents.to(device=restored_residual.device, dtype=torch.float32)
    else:
        reduced_residual = _decode_quantized_block(tail_payload)
        restored_residual = trilinear_resize(
            reduced_residual.unsqueeze(0),
            (original_shape[1], original_shape[2], original_shape[3]),
        ).squeeze(0)
        anchor_source = decoded_anchor_latents.float()
    anchor_reference = anchor_source[:, -1:].expand(-1, original_shape[1], -1, -1)
    return (anchor_reference + restored_residual.float()).contiguous()


def estimate_anchor_plus_tail_codec_bytes(codec_payload: Dict[str, object]) -> int:
    breakdown = estimate_anchor_plus_tail_codec_byte_breakdown(codec_payload)
    return int(sum(int(value) for value in breakdown.values()))


def encode_anchor_plus_tail_latents(
    clean_full_latents: torch.Tensor,
    codec_config,
    chunk_lengths: Sequence[int],
    total_pixels: Optional[int] = None,
    learned_tail_codec=None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
    train_mixed_dual_tail_spatial_factor: Optional[int] = None,
    train_mixed_dual_tail_ratio: float = 0.0,
) -> Dict[str, object]:
    validate_codec_config(codec_config)
    if clean_full_latents.ndim != 4:
        raise ValueError(
            f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}."
        )

    encoded_global_keyframe, decoded_global_keyframe = encode_global_keyframe(
        clean_full_latents[:, :1].contiguous(),
        codec_config,
        move_to_cpu=move_to_cpu,
    )
    global_keyframe_bytes = _estimate_global_keyframe_bytes_from_encoded_entry(encoded_global_keyframe)
    section_ranges = make_section_ranges(
        total_latent_frames=int(clean_full_latents.shape[1]),
        section_span_latents=_as_int(codec_config, "section_span_latents"),
        start_index=1,
    )

    target_low_bpp = _as_optional_float(codec_config, "target_low_bpp")
    decoded_history = decoded_global_keyframe.float().contiguous()
    sections_since_refresh = 0
    section_analyses: List[Dict[str, object]] = []
    for section_start, section_end in section_ranges:
        section_latents = clean_full_latents[:, section_start:section_end]
        section_analysis = _analyze_section_mode_candidates(
            section_latents=section_latents,
            history_latents=decoded_history,
            codec_config=codec_config,
            sections_since_refresh=sections_since_refresh,
            move_to_cpu=move_to_cpu,
        )
        section_analyses.append(section_analysis)

        baseline_mode = str(section_analysis["minimum_mode"])
        decoded_section = _build_proxy_section_from_history(
            history_latents=decoded_history,
            section_shape=tuple(int(value) for value in section_latents.shape),
            anchor_blocks=section_analysis["decoded_blocks_by_mode"][baseline_mode],
        )
        decoded_history = torch.cat([decoded_history, decoded_section.float()], dim=1).contiguous()
        sections_since_refresh = 0 if baseline_mode != PREDICT_ONLY_SECTION_MODE else sections_since_refresh + 1

    if target_low_bpp is not None:
        budget_result = _allocate_section_modes_with_budget(
            section_analyses=section_analyses,
            codec_config=codec_config,
            global_keyframe_bytes=global_keyframe_bytes,
            chunk_lengths=chunk_lengths,
            section_ranges=section_ranges,
            total_pixels=0 if total_pixels is None else int(total_pixels),
        )
        selected_modes = list(budget_result["selected_modes"])
        mode_selection_strategy = "budget"
    else:
        selected_modes = [
            _select_section_mode_from_analysis(section_analysis, codec_config)
            for section_analysis in section_analyses
        ]
        budget_result = {
            "selected_modes": selected_modes,
            "target_low_bpp": None,
            "target_low_bytes": None,
            "estimated_selected_bytes": None,
            "estimated_fixed_bytes": None,
        }
        mode_selection_strategy = "threshold"

    adaptive_full_head_indices = _select_adaptive_dual_head_full_indices(
        section_analyses=section_analyses,
        selected_modes=selected_modes,
        codec_config=codec_config,
    )
    adaptive_soft_pool_hard_indices = _select_adaptive_dual_head_soft_pool_hard_indices(
        section_analyses=section_analyses,
        selected_modes=selected_modes,
        codec_config=codec_config,
    )
    section_payloads: List[Dict[str, object]] = []
    cheap_tail_factor = (
        None
        if train_mixed_dual_tail_spatial_factor is None
        else int(train_mixed_dual_tail_spatial_factor)
    )
    cheap_tail_ratio = float(train_mixed_dual_tail_ratio)
    dual_head_spatial_factor = _dual_head_anchor_spatial_factor(codec_config)
    base_head_spatial_factor = _as_int(codec_config, "anchor_spatial_factor")
    selected_decoded_history = decoded_global_keyframe.float().contiguous()
    selected_sections_since_refresh = 0
    for section_index, ((section_start, section_end), selected_mode) in enumerate(zip(section_ranges, selected_modes)):
        section_latents = clean_full_latents[:, section_start:section_end]
        section_shape = tuple(int(value) for value in section_latents.shape)
        section_analysis = _analyze_section_mode_candidates(
            section_latents=section_latents,
            history_latents=selected_decoded_history,
            codec_config=codec_config,
            sections_since_refresh=selected_sections_since_refresh,
            move_to_cpu=move_to_cpu,
        )
        encoded_anchor_blocks = list(section_analysis["encoded_blocks_by_mode"][selected_mode])
        single_head_fidelity = None
        dual_head_fidelity = None
        dual_tail_fidelity = None
        if selected_mode == SINGLE_REFRESH_SECTION_MODE:
            single_head_fidelity = (
                f"soft_pool_x{_single_head_soft_pool_spatial_factor(codec_config)}"
                if _single_head_codec_type(codec_config) == SINGLE_HEAD_SOFT_POOL_CODEC_TYPE
                else "full"
            )
        if selected_mode == DUAL_REFRESH_SECTION_MODE:
            dual_head_fidelity = "full"
            if _dual_head_codec_type(codec_config) == DUAL_HEAD_P_DELTA_CODEC_TYPE:
                dual_head_fidelity = f"p_delta_x{_dual_head_p_delta_spatial_factor(codec_config)}"
            elif _dual_head_codec_type(codec_config) == DUAL_HEAD_SOFT_POOL_CODEC_TYPE:
                dual_head_fidelity = f"soft_pool_x{_dual_head_soft_pool_spatial_factor(codec_config)}"
                hard_factor = _adaptive_dual_head_soft_pool_hard_factor(codec_config)
                if (
                    hard_factor is not None
                    and section_index in adaptive_soft_pool_hard_indices
                    and len(encoded_anchor_blocks) > 0
                    and len(section_analysis["clean_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 0
                ):
                    head_block_start, head_clean_latents = section_analysis["clean_blocks_by_mode"][
                        DUAL_REFRESH_SECTION_MODE
                    ][0]
                    hard_head_payload = encode_head_soft_pool(
                        head_clean_latents,
                        codec_config,
                        move_to_cpu=move_to_cpu,
                        spatial_factor_override=hard_factor,
                    )
                    old_head_block = encoded_anchor_blocks[0]
                    encoded_anchor_blocks = list(encoded_anchor_blocks)
                    encoded_anchor_blocks[0] = {
                        "start": int(head_block_start),
                        "length": int(head_clean_latents.shape[1]),
                        "payload": hard_head_payload,
                        "bytes": _estimate_refresh_anchor_bytes(hard_head_payload),
                        "head_codec_type": DUAL_HEAD_SOFT_POOL_CODEC_TYPE,
                    }
                    dual_head_fidelity = f"soft_pool_x{hard_factor}"
                    if budget_result["estimated_selected_bytes"] is not None:
                        budget_result["estimated_selected_bytes"] = int(
                            budget_result["estimated_selected_bytes"]
                            + int(encoded_anchor_blocks[0]["bytes"])
                            - int(old_head_block["bytes"])
                        )
            elif (
                dual_head_spatial_factor is not None
                and dual_head_spatial_factor != base_head_spatial_factor
                and len(encoded_anchor_blocks) > 0
            ):
                dual_head_fidelity = f"cheap_x{dual_head_spatial_factor}"
                if section_index in adaptive_full_head_indices:
                    full_head_block = dict(section_analysis["full_head_encoded_block"])
                    cheap_head_block = encoded_anchor_blocks[0]
                    encoded_anchor_blocks = list(encoded_anchor_blocks)
                    encoded_anchor_blocks[0] = full_head_block
                    dual_head_fidelity = "full"
                    if budget_result["estimated_selected_bytes"] is not None:
                        budget_result["estimated_selected_bytes"] = int(
                            budget_result["estimated_selected_bytes"]
                            + int(full_head_block["bytes"])
                            - int(cheap_head_block["bytes"])
                        )
            dual_tail_fidelity = (
                f"p_delta_x{_dual_tail_p_delta_spatial_factor(codec_config)}"
                if _dual_tail_codec_type(codec_config) == DUAL_TAIL_P_DELTA_CODEC_TYPE
                else "full"
            )
            if (
                cheap_tail_factor is not None
                and cheap_tail_factor > 1
                and cheap_tail_ratio > 0.0
                and len(encoded_anchor_blocks) > 1
                and encoded_anchor_blocks[-1].get("tail_codec_type", DUAL_TAIL_ANCHOR_CODEC_TYPE)
                == DUAL_TAIL_ANCHOR_CODEC_TYPE
                and len(section_analysis["clean_blocks_by_mode"][DUAL_REFRESH_SECTION_MODE]) > 1
                and float(torch.rand((), device=clean_full_latents.device).item()) < cheap_tail_ratio
            ):
                tail_block_start, tail_clean_latents = section_analysis["clean_blocks_by_mode"][
                    DUAL_REFRESH_SECTION_MODE
                ][-1]
                cheap_tail_payload = encode_refresh_anchor(
                    tail_clean_latents,
                    codec_config,
                    move_to_cpu=move_to_cpu,
                    spatial_factor_override=cheap_tail_factor,
                )
                encoded_anchor_blocks = list(encoded_anchor_blocks)
                encoded_anchor_blocks[-1] = {
                    "start": int(tail_block_start),
                    "length": int(tail_clean_latents.shape[1]),
                    "payload": cheap_tail_payload,
                    "bytes": _estimate_refresh_anchor_bytes(cheap_tail_payload),
                }
                dual_tail_fidelity = f"cheap_x{cheap_tail_factor}"
        decoded_anchor_blocks = _decode_section_anchor_blocks_from_payload(
            anchor_blocks=encoded_anchor_blocks,
            decoded_history=selected_decoded_history,
            section_shape=section_shape,
            target_device=selected_decoded_history.device,
        )
        decoded_section = _build_proxy_section_from_history(
            history_latents=selected_decoded_history,
            section_shape=section_shape,
            anchor_blocks=decoded_anchor_blocks,
        )
        selected_decoded_history = torch.cat(
            [selected_decoded_history, decoded_section.float()],
            dim=1,
        ).contiguous()
        selected_sections_since_refresh = (
            0 if selected_mode != PREDICT_ONLY_SECTION_MODE else selected_sections_since_refresh + 1
        )
        section_payloads.append(
            {
                "mode": str(selected_mode),
                "anchor_blocks": encoded_anchor_blocks,
                "heuristic_scores": dict(section_analysis["heuristic_scores"]),
                "single_head_fidelity": single_head_fidelity,
                "dual_head_fidelity": dual_head_fidelity,
                "dual_tail_fidelity": dual_tail_fidelity,
            }
        )

    return {
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "codec_config": dict(codec_config) if isinstance(codec_config, dict) else None,
        "global_keyframe_codec_type": str(encoded_global_keyframe["codec_type"]),
        "global_keyframe": encoded_global_keyframe.get("global_keyframe"),
        "global_keyframe_payload": encoded_global_keyframe.get("global_keyframe_payload"),
        "section_ranges": [(int(start), int(end)) for start, end in section_ranges],
        "section_payloads": section_payloads,
        "mode_selection_strategy": mode_selection_strategy,
        "target_low_bpp": budget_result["target_low_bpp"],
        "target_low_bytes": budget_result["target_low_bytes"],
        "estimated_selected_bytes": budget_result["estimated_selected_bytes"],
        "estimated_fixed_bytes": budget_result["estimated_fixed_bytes"],
    }


def _decode_section_anchor_blocks_from_payload(
    anchor_blocks: Sequence[Dict[str, object]],
    decoded_history: torch.Tensor,
    section_shape: Tuple[int, int, int, int],
    target_device: torch.device,
) -> List[Tuple[int, torch.Tensor]]:
    channel_count, section_length, height, width = (int(value) for value in section_shape)
    decoded_anchor_blocks: List[Tuple[int, torch.Tensor]] = []
    head_p_delta_blocks: List[Dict[str, object]] = []
    head_soft_pool_blocks: List[Dict[str, object]] = []
    tail_p_delta_blocks: List[Dict[str, object]] = []
    for anchor_block in anchor_blocks:
        anchor_payload = anchor_block["payload"]
        block_codec_type = str(
            anchor_block.get(
                "head_codec_type",
                anchor_block.get(
                    "tail_codec_type",
                    anchor_payload.get("codec_type", DUAL_TAIL_ANCHOR_CODEC_TYPE),
                ),
            )
        )
        if block_codec_type == DUAL_HEAD_SOFT_POOL_CODEC_TYPE:
            head_soft_pool_blocks.append(anchor_block)
            continue
        if block_codec_type == DUAL_HEAD_P_DELTA_CODEC_TYPE:
            head_p_delta_blocks.append(anchor_block)
            continue
        if block_codec_type == DUAL_TAIL_P_DELTA_CODEC_TYPE:
            tail_p_delta_blocks.append(anchor_block)
            continue
        decoded_anchor = decode_refresh_anchor(anchor_payload).to(device=target_device, dtype=torch.float32)
        decoded_anchor_blocks.append((int(anchor_block["start"]), decoded_anchor))
    for anchor_block in sorted(head_soft_pool_blocks, key=lambda block: int(block["start"])):
        block_start = int(anchor_block["start"])
        decoded_head = decode_head_soft_pool(anchor_block["payload"]).to(
            device=target_device,
            dtype=torch.float32,
        )
        decoded_anchor_blocks.append((block_start, decoded_head))
    for anchor_block in sorted(head_p_delta_blocks, key=lambda block: int(block["start"])):
        block_start = int(anchor_block["start"])
        block_length = int(anchor_block["length"])
        prediction_without_head = _build_proxy_section_from_history(
            history_latents=decoded_history,
            section_shape=(channel_count, section_length, height, width),
            anchor_blocks=decoded_anchor_blocks,
        ).to(device=target_device, dtype=torch.float32)
        predicted_head = prediction_without_head[:, block_start : block_start + block_length].contiguous()
        decoded_head = decode_head_p_delta(anchor_block["payload"], predicted_head).to(
            device=target_device,
            dtype=torch.float32,
        )
        decoded_anchor_blocks.append((block_start, decoded_head))
    for anchor_block in sorted(tail_p_delta_blocks, key=lambda block: int(block["start"])):
        block_start = int(anchor_block["start"])
        block_length = int(anchor_block["length"])
        prediction_without_tail = _build_proxy_section_from_history(
            history_latents=decoded_history,
            section_shape=(channel_count, section_length, height, width),
            anchor_blocks=decoded_anchor_blocks,
        ).to(device=target_device, dtype=torch.float32)
        predicted_tail = prediction_without_tail[:, block_start : block_start + block_length].contiguous()
        decoded_tail = decode_tail_p_delta(anchor_block["payload"], predicted_tail).to(
            device=target_device,
            dtype=torch.float32,
        )
        decoded_anchor_blocks.append((block_start, decoded_tail))
    return decoded_anchor_blocks


def decode_anchor_plus_tail_latents(codec_payload: Dict[str, object], learned_tail_codec=None) -> torch.Tensor:
    global_keyframe = decode_global_keyframe(codec_payload).float()
    target_device = _module_device(learned_tail_codec) or global_keyframe.device
    global_keyframe = global_keyframe.to(device=target_device, dtype=torch.float32)
    section_ranges = [tuple(int(value) for value in section_range) for section_range in codec_payload["section_ranges"]]

    if not section_ranges:
        return global_keyframe.float().contiguous()

    if codec_payload.get("section_payloads"):
        decoded_history = global_keyframe.float().contiguous()
        channel_count = int(global_keyframe.shape[0])
        height = int(global_keyframe.shape[2])
        width = int(global_keyframe.shape[3])
        for (section_start, section_end), section_payload in zip(section_ranges, codec_payload["section_payloads"]):
            section_length = int(section_end) - int(section_start)
            decoded_anchor_blocks = _decode_section_anchor_blocks_from_payload(
                anchor_blocks=section_payload.get("anchor_blocks", []),
                decoded_history=decoded_history,
                section_shape=(channel_count, section_length, height, width),
                target_device=target_device,
            )
            decoded_section = _build_proxy_section_from_history(
                history_latents=decoded_history,
                section_shape=(channel_count, section_length, height, width),
                anchor_blocks=decoded_anchor_blocks,
            ).to(device=target_device, dtype=torch.float32)
            decoded_history = torch.cat([decoded_history, decoded_section], dim=1).contiguous()
        return decoded_history.float().contiguous()

    section_anchor_payloads = codec_payload["section_anchor_payloads"]
    section_tail_payloads = codec_payload["section_tail_payloads"]
    sections: List[torch.Tensor] = []
    for (section_start, section_end), anchor_payload, tail_payload in zip(
        section_ranges, section_anchor_payloads, section_tail_payloads
    ):
        del section_start, section_end
        decoded_anchor = decode_refresh_anchor(anchor_payload)
        decoded_tail = decode_section_tail_residual(
            tail_payload,
            decoded_anchor,
            learned_tail_codec=learned_tail_codec,
        )
        decoded_anchor = decoded_anchor.to(device=target_device, dtype=torch.float32)
        decoded_tail = decoded_tail.to(device=target_device, dtype=torch.float32)
        sections.append(torch.cat([decoded_anchor, decoded_tail], dim=1))

    remainder = torch.cat(sections, dim=1)
    return torch.cat([global_keyframe, remainder.float()], dim=1).contiguous()


def decode_legacy_low_latents(codec_payload: Dict[str, object]) -> torch.Tensor:
    keyframe = codec_payload["keyframe"].float()
    quantized_remainder = codec_payload["quantized_remainder"]
    original_remainder_shape = tuple(int(value) for value in codec_payload["original_remainder_shape"])
    if isinstance(quantized_remainder, torch.Tensor) and quantized_remainder.numel() == 0:
        return keyframe.float().contiguous()

    reduced_remainder = symmetric_dequantize_per_channel(
        quantized_latents=codec_payload["quantized_remainder"],
        scales=codec_payload["scales"],
    )
    restored_remainder = trilinear_resize(
        reduced_remainder.unsqueeze(0),
        (original_remainder_shape[1], original_remainder_shape[2], original_remainder_shape[3]),
    ).squeeze(0)
    return torch.cat([keyframe.float(), restored_remainder.float()], dim=1).contiguous()


def decode_low_latents_payload(codec_payload: Dict[str, object], learned_tail_codec=None) -> torch.Tensor:
    if codec_payload.get("section_payloads"):
        return decode_anchor_plus_tail_latents(codec_payload, learned_tail_codec=learned_tail_codec)
    if "section_anchor_payloads" in codec_payload and "section_tail_payloads" in codec_payload:
        return decode_anchor_plus_tail_latents(codec_payload, learned_tail_codec=learned_tail_codec)
    if "quantized_remainder" in codec_payload and "keyframe" in codec_payload:
        return decode_legacy_low_latents(codec_payload)
    raise KeyError("Unrecognized low codec payload. Expected either anchor+tail fields or legacy remainder fields.")
