import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from reconstruct.learned_codec import LEARNED_TAIL_CODEC_TYPE


EPS = 1e-8
TRILINEAR_TAIL_CODEC_TYPE = "trilinear"
RAW_KEYFRAME_CODEC_MODE = "raw"
QUANTIZED_INT8_KEYFRAME_CODEC_MODE = "quantized_int8"


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


def _as_float(codec_config, key: str) -> float:
    return float(_get_config_value(codec_config, key))


def _as_optional_float(codec_config, key: str, default: float | None = None) -> float | None:
    value = _get_optional_config_value(codec_config, key, default)
    if value is None:
        return None
    return float(value)


def _as_bool(codec_config, key: str, default: bool = False) -> bool:
    value = _get_optional_config_value(codec_config, key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tail_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    return str(getattr(codec_config, "tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))


def _keyframe_codec_mode(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("keyframe_codec_mode", RAW_KEYFRAME_CODEC_MODE))
    return str(getattr(codec_config, "keyframe_codec_mode", RAW_KEYFRAME_CODEC_MODE))


def _dynamic_rate_policy(codec_config) -> Dict[str, object]:
    return {
        "enabled": _as_bool(codec_config, "dynamic_rate_enabled", False),
        "motion_score_type": str(_get_optional_config_value(codec_config, "motion_score_type", "latent_delta_l1_norm")),
        "motion_q20": _as_optional_float(codec_config, "motion_q20"),
        "motion_q80": _as_optional_float(codec_config, "motion_q80"),
        "tail_quality_min": float(_get_optional_config_value(codec_config, "tail_quality_min", 0.75)),
        "tail_quality_max": float(_get_optional_config_value(codec_config, "tail_quality_max", 1.35)),
        "anchor_profile_table": {
            "low": int(_get_optional_config_value(codec_config, "anchor_profile_low", 4)),
            "base": int(_get_optional_config_value(codec_config, "anchor_profile_base", 2)),
            "high": int(_get_optional_config_value(codec_config, "anchor_profile_high", 1)),
        },
    }


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
    for key in ("quant_dtype", "anchor_quant_dtype"):
        value = _as_str(codec_config, key)
        if value != "int8":
            raise ValueError(f"{key}={value} is unsupported. Only int8 is implemented.")
    keyframe_dtype = _as_str(codec_config, "keyframe_dtype")
    if keyframe_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported keyframe_dtype={keyframe_dtype}.")
    keyframe_codec_mode = _keyframe_codec_mode(codec_config)
    if keyframe_codec_mode not in {RAW_KEYFRAME_CODEC_MODE, QUANTIZED_INT8_KEYFRAME_CODEC_MODE}:
        raise ValueError(
            f"Unsupported keyframe_codec_mode={keyframe_codec_mode}. "
            f"Expected {RAW_KEYFRAME_CODEC_MODE} or {QUANTIZED_INT8_KEYFRAME_CODEC_MODE}."
        )
    keyframe_quant_dtype = _as_str(codec_config, "keyframe_quant_dtype")
    if keyframe_quant_dtype != "int8":
        raise ValueError(f"keyframe_quant_dtype={keyframe_quant_dtype} is unsupported. Only int8 is implemented.")
    keyframe_spatial_factor = _as_int(codec_config, "keyframe_spatial_factor")
    if keyframe_spatial_factor < 1:
        raise ValueError(f"keyframe_spatial_factor must be >= 1, got {keyframe_spatial_factor}.")
    tail_codec_type = _tail_codec_type(codec_config)
    if tail_codec_type not in {TRILINEAR_TAIL_CODEC_TYPE, LEARNED_TAIL_CODEC_TYPE}:
        raise ValueError(
            f"Unsupported tail_codec_type={tail_codec_type}. "
            f"Expected {TRILINEAR_TAIL_CODEC_TYPE} or {LEARNED_TAIL_CODEC_TYPE}."
        )
    dynamic_rate_enabled = _as_bool(codec_config, "dynamic_rate_enabled", False)
    motion_score_type = str(_get_optional_config_value(codec_config, "motion_score_type", "latent_delta_l1_norm"))
    if motion_score_type != "latent_delta_l1_norm":
        raise ValueError(
            f"Unsupported motion_score_type={motion_score_type}. Only latent_delta_l1_norm is implemented."
        )
    tail_quality_min = float(_get_optional_config_value(codec_config, "tail_quality_min", 0.75))
    tail_quality_max = float(_get_optional_config_value(codec_config, "tail_quality_max", 1.35))
    if tail_quality_min <= 0.0 or tail_quality_max <= 0.0:
        raise ValueError(
            f"tail_quality_min and tail_quality_max must both be > 0, got {tail_quality_min}, {tail_quality_max}."
        )
    if tail_quality_max < tail_quality_min:
        raise ValueError(
            f"tail_quality_max must be >= tail_quality_min, got {tail_quality_max} < {tail_quality_min}."
        )
    for key in ("anchor_profile_low", "anchor_profile_base", "anchor_profile_high"):
        value = int(_get_optional_config_value(codec_config, key, {"anchor_profile_low": 4, "anchor_profile_base": 2, "anchor_profile_high": 1}[key]))
        if value < 1:
            raise ValueError(f"{key} must be >= 1, got {value}.")
    if dynamic_rate_enabled:
        motion_q20 = _as_optional_float(codec_config, "motion_q20")
        motion_q80 = _as_optional_float(codec_config, "motion_q80")
        if motion_q20 is not None and motion_q80 is not None and motion_q80 < motion_q20:
            raise ValueError(f"motion_q80 must be >= motion_q20, got {motion_q80} < {motion_q20}.")


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


def compute_section_motion_scores(
    clean_full_latents: torch.Tensor,
    section_ranges: Sequence[Tuple[int, int]],
) -> List[float]:
    if clean_full_latents.ndim != 4:
        raise ValueError(
            f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}."
        )
    motion_scores: List[float] = []
    for section_start, section_end in section_ranges:
        if section_end <= section_start:
            motion_scores.append(0.0)
            continue
        current_section = clean_full_latents[:, section_start:section_end].float()
        previous_frame = clean_full_latents[:, section_start - 1 : section_start].float()
        section_with_prev = torch.cat([previous_frame, current_section], dim=1)
        delta = section_with_prev[:, 1:] - section_with_prev[:, :-1]
        numerator = delta.abs().mean()
        denominator = current_section.abs().mean() + EPS
        motion_scores.append(float((numerator / denominator).detach().cpu().item()))
    return motion_scores


def _motion_interpolation_weight(motion_score: float, motion_q20: float, motion_q80: float) -> float:
    if motion_q80 <= motion_q20 + EPS:
        return 1.0 if motion_score > motion_q20 else 0.0
    return float(min(1.0, max(0.0, (motion_score - motion_q20) / (motion_q80 - motion_q20))))


def _resolve_section_anchor_profile(motion_score: float, codec_config) -> Tuple[str, int]:
    profile_table = _dynamic_rate_policy(codec_config)["anchor_profile_table"]
    motion_q20 = _as_optional_float(codec_config, "motion_q20")
    motion_q80 = _as_optional_float(codec_config, "motion_q80")
    if motion_q20 is None or motion_q80 is None:
        raise ValueError("Dynamic rate allocation requires motion_q20 and motion_q80 to be calibrated.")
    if motion_score <= motion_q20:
        return "low", int(profile_table["low"])
    if motion_score >= motion_q80:
        return "high", int(profile_table["high"])
    return "base", int(profile_table["base"])


def _resolve_section_tail_quality_scale(motion_score: float, codec_config) -> float:
    motion_q20 = _as_optional_float(codec_config, "motion_q20")
    motion_q80 = _as_optional_float(codec_config, "motion_q80")
    if motion_q20 is None or motion_q80 is None:
        raise ValueError("Dynamic rate allocation requires motion_q20 and motion_q80 to be calibrated.")
    quality_min = float(_get_optional_config_value(codec_config, "tail_quality_min", 0.75))
    quality_max = float(_get_optional_config_value(codec_config, "tail_quality_max", 1.35))
    weight = _motion_interpolation_weight(motion_score, motion_q20, motion_q80)
    return float(quality_min + weight * (quality_max - quality_min))


def _estimate_tensor_payload_bytes(payload: Dict[str, object]) -> int:
    total_bytes = 0
    for tensor_key in ("quantized", "scales"):
        value = payload.get(tensor_key)
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()
    for bytes_key in ("y_string", "z_string"):
        if bytes_key in payload:
            total_bytes += len(bytes(payload[bytes_key]))
    if "quality_side_data" in payload:
        total_bytes += len(bytes(payload["quality_side_data"]))
    elif "quality_side_bytes" in payload:
        total_bytes += int(payload["quality_side_bytes"])
    if "reconstructed_residual" in payload:
        rate_bits = payload.get("rate_bits")
        if torch.is_tensor(rate_bits):
            total_bytes += int(math.ceil(float(rate_bits.detach().cpu().item()) / 8.0))
        else:
            total_bytes += int(math.ceil(float(rate_bits or 0.0) / 8.0))
    for shape_key in ("reduced_shape", "original_shape", "y_shape", "z_shape"):
        total_bytes += len(payload.get(shape_key, [])) * 4
    return int(total_bytes)


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


def encode_refresh_anchor(anchor_latents: torch.Tensor, codec_config, move_to_cpu: bool = True) -> Dict[str, object]:
    if anchor_latents.ndim != 4:
        raise ValueError(f"Expected anchor latents with shape [C, T, H, W], got {tuple(anchor_latents.shape)}.")
    if anchor_latents.shape[1] == 0:
        raise ValueError("Anchor latents must contain at least one time step.")

    anchor_spatial_factor = _as_int(codec_config, "anchor_spatial_factor")
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


def encode_global_keyframe(global_keyframe: torch.Tensor, codec_config, move_to_cpu: bool = True):
    if global_keyframe.ndim != 4:
        raise ValueError(f"Expected global_keyframe with shape [C, T, H, W], got {tuple(global_keyframe.shape)}.")
    if global_keyframe.shape[1] != 1:
        raise ValueError(f"Expected a single-frame global keyframe, got shape {tuple(global_keyframe.shape)}.")

    keyframe_codec_mode = _keyframe_codec_mode(codec_config)
    if keyframe_codec_mode == RAW_KEYFRAME_CODEC_MODE:
        keyframe_dtype = torch.float16 if _as_str(codec_config, "keyframe_dtype") == "float16" else torch.float32
        return _maybe_to_cpu(global_keyframe.to(dtype=keyframe_dtype), move_to_cpu)

    reduced_keyframe = global_keyframe
    keyframe_spatial_factor = _as_int(codec_config, "keyframe_spatial_factor")
    if keyframe_spatial_factor > 1:
        reduced_keyframe = trilinear_resize(
            global_keyframe.unsqueeze(0),
            (
                global_keyframe.shape[1],
                max(1, math.ceil(global_keyframe.shape[2] / keyframe_spatial_factor)),
                max(1, math.ceil(global_keyframe.shape[3] / keyframe_spatial_factor)),
            ),
        ).squeeze(0)
    payload = _encode_quantized_block(
        reduced_keyframe,
        _as_str(codec_config, "keyframe_quant_dtype"),
        move_to_cpu=move_to_cpu,
    )
    payload["original_shape"] = [int(value) for value in global_keyframe.shape]
    payload["keyframe_codec_mode"] = keyframe_codec_mode
    payload["quant_dtype"] = _as_str(codec_config, "keyframe_quant_dtype")
    return payload


def decode_global_keyframe(global_keyframe_payload) -> torch.Tensor:
    if isinstance(global_keyframe_payload, torch.Tensor):
        return global_keyframe_payload.float().contiguous()
    if not isinstance(global_keyframe_payload, dict):
        raise TypeError(f"Unsupported global_keyframe payload type: {type(global_keyframe_payload)!r}.")

    original_shape = tuple(int(value) for value in global_keyframe_payload["original_shape"])
    reduced_keyframe = _decode_quantized_block(global_keyframe_payload)
    if tuple(reduced_keyframe.shape) == original_shape:
        return reduced_keyframe.float().contiguous()
    restored_keyframe = trilinear_resize(
        reduced_keyframe.unsqueeze(0),
        (original_shape[1], original_shape[2], original_shape[3]),
    ).squeeze(0)
    return restored_keyframe.float().contiguous()


def encode_section_tail_residual(
    section_tail_latents: torch.Tensor,
    decoded_anchor_latents: torch.Tensor,
    codec_config,
    learned_tail_codec=None,
    quality_scale: float | torch.Tensor | None = None,
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
            "quality_scale": float(1.0 if quality_scale is None else float(quality_scale)),
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
        anchor_condition = decoded_anchor_latents.unsqueeze(0)
        if use_ste_quant:
            payload = learned_tail_codec.encode_to_training_payload(
                residual=residual_tail.unsqueeze(0),
                anchor_latents=anchor_condition,
                quality_scale=quality_scale,
            )
        else:
            payload = learned_tail_codec.encode_to_bitstream_payload(
                residual=residual_tail.unsqueeze(0),
                anchor_latents=anchor_condition,
                quality_scale=quality_scale,
            )
        payload["tail_codec_type"] = LEARNED_TAIL_CODEC_TYPE
        payload["original_shape"] = original_shape
        payload["quality_scale"] = float(1.0 if quality_scale is None else float(quality_scale))
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
    payload["quality_scale"] = float(1.0 if quality_scale is None else float(quality_scale))
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
        restored_residual = learned_tail_codec.decode_payload(
            tail_payload,
            anchor_latents=decoded_anchor_latents.unsqueeze(0),
            quality_scale=tail_payload.get("quality_scale"),
        )
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
    total_bytes = 0
    global_keyframe = codec_payload.get("global_keyframe")
    if isinstance(global_keyframe, torch.Tensor):
        total_bytes += global_keyframe.numel() * global_keyframe.element_size()
    elif isinstance(global_keyframe, dict):
        for tensor_key in ("quantized", "scales"):
            value = global_keyframe.get(tensor_key)
            if isinstance(value, torch.Tensor):
                total_bytes += value.numel() * value.element_size()
        for shape_key in ("reduced_shape", "original_shape"):
            total_bytes += len(global_keyframe.get(shape_key, [])) * 4

    for payload_key in ("section_anchor_payloads", "section_tail_payloads"):
        for payload in codec_payload.get(payload_key, []):
            for tensor_key in ("quantized", "scales"):
                value = payload.get(tensor_key)
                if isinstance(value, torch.Tensor):
                    total_bytes += value.numel() * value.element_size()
            if "reconstructed_residual" in payload:
                rate_bits = payload.get("rate_bits")
                if torch.is_tensor(rate_bits):
                    total_bytes += int(math.ceil(float(rate_bits.detach().cpu().item()) / 8.0))
                else:
                    total_bytes += int(math.ceil(float(rate_bits or 0.0) / 8.0))
            if "y_string" in payload:
                total_bytes += len(bytes(payload["y_string"]))
            if "z_string" in payload:
                total_bytes += len(bytes(payload["z_string"]))
            if "quality_side_data" in payload:
                total_bytes += len(bytes(payload["quality_side_data"]))
            elif "quality_side_bytes" in payload:
                total_bytes += int(payload["quality_side_bytes"])

    metadata_values: List[int] = [len(codec_payload.get("chunk_lengths", []))]
    metadata_values.extend(int(value) for value in codec_payload.get("chunk_lengths", []))
    for start, end in codec_payload.get("section_ranges", []):
        metadata_values.extend([int(start), int(end)])
    total_bytes += len(metadata_values) * 4

    for payload_key in ("section_anchor_payloads", "section_tail_payloads"):
        for payload in codec_payload.get(payload_key, []):
            for shape_key in ("reduced_shape", "original_shape", "y_shape", "z_shape"):
                total_bytes += len(payload.get(shape_key, [])) * 4

    return int(total_bytes)


def encode_anchor_plus_tail_latents(
    clean_full_latents: torch.Tensor,
    codec_config,
    chunk_lengths: Sequence[int],
    learned_tail_codec=None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
) -> Dict[str, object]:
    validate_codec_config(codec_config)
    if clean_full_latents.ndim != 4:
        raise ValueError(
            f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}."
        )

    global_keyframe = encode_global_keyframe(
        clean_full_latents[:, :1].contiguous(),
        codec_config,
        move_to_cpu=move_to_cpu,
    )
    section_ranges = make_section_ranges(
        total_latent_frames=int(clean_full_latents.shape[1]),
        section_span_latents=_as_int(codec_config, "section_span_latents"),
        start_index=1,
    )
    dynamic_rate_enabled = _as_bool(codec_config, "dynamic_rate_enabled", False)
    section_motion_scores = compute_section_motion_scores(clean_full_latents, section_ranges)

    section_anchor_payloads: List[Dict[str, object]] = []
    section_tail_payloads: List[Dict[str, object]] = []
    section_anchor_profiles: List[str] = []
    section_tail_quality_scales: List[float] = []
    section_anchor_bytes: List[int] = []
    section_tail_bytes: List[int] = []
    section_total_bytes: List[int] = []
    aggregated_rate_bits: List[torch.Tensor] = []
    aggregated_y_bits: List[torch.Tensor] = []
    aggregated_z_bits: List[torch.Tensor] = []
    for section_idx, (section_start, section_end) in enumerate(section_ranges):
        section_latents = clean_full_latents[:, section_start:section_end]
        anchor_span = min(_as_int(codec_config, "anchor_span_latents"), section_latents.shape[1])
        section_anchor = section_latents[:, :anchor_span].contiguous()
        section_tail = section_latents[:, anchor_span:].contiguous()
        section_codec_config = dict(codec_config) if isinstance(codec_config, dict) else {
            "temporal_factor": _as_int(codec_config, "temporal_factor"),
            "spatial_factor": _as_int(codec_config, "spatial_factor"),
            "quant_dtype": _as_str(codec_config, "quant_dtype"),
            "keyframe_dtype": _as_str(codec_config, "keyframe_dtype"),
            "keyframe_codec_mode": _keyframe_codec_mode(codec_config),
            "keyframe_quant_dtype": _as_str(codec_config, "keyframe_quant_dtype"),
            "keyframe_spatial_factor": _as_int(codec_config, "keyframe_spatial_factor"),
            "section_span_latents": _as_int(codec_config, "section_span_latents"),
            "anchor_span_latents": _as_int(codec_config, "anchor_span_latents"),
            "tail_span_latents": _as_int(codec_config, "tail_span_latents"),
            "anchor_quant_dtype": _as_str(codec_config, "anchor_quant_dtype"),
            "anchor_spatial_factor": _as_int(codec_config, "anchor_spatial_factor"),
            "tail_codec_type": _tail_codec_type(codec_config),
            "dynamic_rate_enabled": dynamic_rate_enabled,
            "motion_score_type": str(_get_optional_config_value(codec_config, "motion_score_type", "latent_delta_l1_norm")),
            "motion_q20": _as_optional_float(codec_config, "motion_q20"),
            "motion_q80": _as_optional_float(codec_config, "motion_q80"),
            "tail_quality_min": float(_get_optional_config_value(codec_config, "tail_quality_min", 0.75)),
            "tail_quality_max": float(_get_optional_config_value(codec_config, "tail_quality_max", 1.35)),
            "anchor_profile_low": int(_get_optional_config_value(codec_config, "anchor_profile_low", 4)),
            "anchor_profile_base": int(_get_optional_config_value(codec_config, "anchor_profile_base", 2)),
            "anchor_profile_high": int(_get_optional_config_value(codec_config, "anchor_profile_high", 1)),
        }
        motion_score = section_motion_scores[section_idx] if section_idx < len(section_motion_scores) else 0.0
        anchor_profile_name = "base"
        tail_quality_scale = 1.0
        if dynamic_rate_enabled:
            anchor_profile_name, anchor_spatial_factor = _resolve_section_anchor_profile(motion_score, codec_config)
            section_codec_config["anchor_spatial_factor"] = int(anchor_spatial_factor)
            tail_quality_scale = _resolve_section_tail_quality_scale(motion_score, codec_config)
        anchor_payload = encode_refresh_anchor(section_anchor, section_codec_config, move_to_cpu=move_to_cpu)
        decoded_anchor = decode_refresh_anchor(anchor_payload)
        tail_payload = encode_section_tail_residual(
            section_tail,
            decoded_anchor,
            section_codec_config,
            learned_tail_codec=learned_tail_codec,
            quality_scale=tail_quality_scale,
            use_ste_quant=use_ste_quant,
            move_to_cpu=move_to_cpu,
        )
        section_anchor_payloads.append(anchor_payload)
        section_tail_payloads.append(tail_payload)
        section_anchor_profiles.append(anchor_profile_name)
        section_tail_quality_scales.append(float(tail_quality_scale))
        anchor_payload_bytes = _estimate_tensor_payload_bytes(anchor_payload)
        tail_payload_bytes = _estimate_tensor_payload_bytes(tail_payload)
        section_anchor_bytes.append(anchor_payload_bytes)
        section_tail_bytes.append(tail_payload_bytes)
        section_total_bytes.append(anchor_payload_bytes + tail_payload_bytes)
        if torch.is_tensor(tail_payload.get("rate_bits")):
            aggregated_rate_bits.append(tail_payload["rate_bits"])
        if torch.is_tensor(tail_payload.get("y_bits")):
            aggregated_y_bits.append(tail_payload["y_bits"])
        if torch.is_tensor(tail_payload.get("z_bits")):
            aggregated_z_bits.append(tail_payload["z_bits"])

    codec_payload = {
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "codec_config": dict(codec_config) if isinstance(codec_config, dict) else None,
        "global_keyframe": global_keyframe,
        "section_ranges": [(int(start), int(end)) for start, end in section_ranges],
        "section_anchor_payloads": section_anchor_payloads,
        "section_tail_payloads": section_tail_payloads,
        "dynamic_rate_policy": _dynamic_rate_policy(codec_config),
        "section_motion_scores": [float(value) for value in section_motion_scores],
        "section_anchor_profiles": section_anchor_profiles,
        "section_tail_quality_scales": [float(value) for value in section_tail_quality_scales],
        "section_anchor_bytes": [int(value) for value in section_anchor_bytes],
        "section_tail_bytes": [int(value) for value in section_tail_bytes],
        "section_total_bytes": [int(value) for value in section_total_bytes],
    }
    if aggregated_rate_bits:
        codec_payload["rate_bits"] = torch.stack(aggregated_rate_bits).sum()
    if aggregated_y_bits:
        codec_payload["y_bits"] = torch.stack(aggregated_y_bits).sum()
    if aggregated_z_bits:
        codec_payload["z_bits"] = torch.stack(aggregated_z_bits).sum()
    return codec_payload


def decode_anchor_plus_tail_latents(codec_payload: Dict[str, object], learned_tail_codec=None) -> torch.Tensor:
    global_keyframe = decode_global_keyframe(codec_payload["global_keyframe"])
    target_device = _module_device(learned_tail_codec) or global_keyframe.device
    global_keyframe = global_keyframe.to(device=target_device, dtype=torch.float32)
    section_ranges = [tuple(int(value) for value in section_range) for section_range in codec_payload["section_ranges"]]
    section_anchor_payloads = codec_payload["section_anchor_payloads"]
    section_tail_payloads = codec_payload["section_tail_payloads"]

    if not section_ranges:
        return global_keyframe.float().contiguous()

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
    if "section_anchor_payloads" in codec_payload and "section_tail_payloads" in codec_payload:
        return decode_anchor_plus_tail_latents(codec_payload, learned_tail_codec=learned_tail_codec)
    if "quantized_remainder" in codec_payload and "keyframe" in codec_payload:
        return decode_legacy_low_latents(codec_payload)
    raise KeyError("Unrecognized low codec payload. Expected either anchor+tail fields or legacy remainder fields.")
