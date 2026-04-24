import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from reconstruct.learned_codec import LEARNED_TAIL_CODEC_TYPE


EPS = 1e-8
TRILINEAR_TAIL_CODEC_TYPE = "trilinear"
PREDICT_ONLY_SECTION_MODE = "predict_only"
SINGLE_REFRESH_SECTION_MODE = "single_refresh"
DUAL_REFRESH_SECTION_MODE = "dual_refresh"
DEFAULT_MAX_PREDICT_ONLY_GAP_SECTIONS = 2
DEFAULT_SINGLE_REFRESH_GAIN_THRESHOLD = 0.12
DEFAULT_DUAL_REFRESH_GAIN_THRESHOLD = 0.18
DEFAULT_BOUNDARY_JUMP_THRESHOLD = 0.18
DEFAULT_CUT_DETECTION_THRESHOLD = 0.35


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


def _dual_tail_anchor_spatial_factor(codec_config) -> Optional[int]:
    value = _get_optional_config_value(codec_config, "dual_tail_anchor_spatial_factor", None)
    if value is None:
        return None
    return int(value)


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
    if dual_tail_anchor_spatial_factor is not None and dual_tail_anchor_spatial_factor < 1:
        raise ValueError(
            "dual_tail_anchor_spatial_factor must be >= 1 when set, "
            f"got {dual_tail_anchor_spatial_factor}."
        )
    for key in ("quant_dtype", "anchor_quant_dtype"):
        value = _as_str(codec_config, key)
        if value != "int8":
            raise ValueError(f"{key}={value} is unsupported. Only int8 is implemented.")
    keyframe_dtype = _as_str(codec_config, "keyframe_dtype")
    if keyframe_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported keyframe_dtype={keyframe_dtype}.")
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

    steps = torch.arange(1, section_length + 1, device=device, dtype=torch.float32).view(1, section_length, 1, 1)
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
    total_bytes = 0
    for tensor_key in ("quantized", "scales"):
        value = anchor_payload.get(tensor_key)
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()
    for shape_key in ("reduced_shape", "original_shape"):
        total_bytes += len(anchor_payload.get(shape_key, [])) * 4
    return int(total_bytes)


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

    head_payload = encode_refresh_anchor(head_clean_block[1], codec_config, move_to_cpu=move_to_cpu)
    head_decoded_block = (
        int(head_clean_block[0]),
        decode_refresh_anchor(head_payload).to(device=section_latents.device, dtype=torch.float32),
    )
    single_decoded_blocks = [head_decoded_block]
    single_encoded_blocks = [
        {
            "start": int(head_clean_block[0]),
            "length": int(head_clean_block[1].shape[1]),
            "payload": head_payload,
            "bytes": _estimate_refresh_anchor_bytes(head_payload),
        }
    ]

    dual_decoded_blocks = list(single_decoded_blocks)
    dual_encoded_blocks = list(single_encoded_blocks)
    if tail_start >= anchor_span and tail_start < section_length:
        tail_payload = encode_refresh_anchor(
            tail_clean_block[1],
            codec_config,
            move_to_cpu=move_to_cpu,
            spatial_factor_override=_dual_tail_anchor_spatial_factor(codec_config),
        )
        dual_decoded_blocks.append(
            (
                int(tail_clean_block[0]),
                decode_refresh_anchor(tail_payload).to(device=section_latents.device, dtype=torch.float32),
            )
        )
        dual_encoded_blocks.append(
            {
                "start": int(tail_clean_block[0]),
                "length": int(tail_clean_block[1].shape[1]),
                "payload": tail_payload,
                "bytes": _estimate_refresh_anchor_bytes(tail_payload),
            }
        )

    none_prediction = _build_proxy_section_from_history(
        history_latents,
        tuple(int(value) for value in section_latents.shape),
        [],
    )
    single_prediction = _build_proxy_section_from_history(
        history_latents,
        tuple(int(value) for value in section_latents.shape),
        single_decoded_blocks,
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


def _estimate_budget_fixed_bytes(
    global_keyframe: torch.Tensor,
    chunk_lengths: Sequence[int],
    section_ranges: Sequence[Tuple[int, int]],
    section_analyses: Sequence[Dict[str, object]],
) -> int:
    total_bytes = 0
    total_bytes += global_keyframe.numel() * global_keyframe.element_size()
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
    global_keyframe: torch.Tensor,
    chunk_lengths: Sequence[int],
    section_ranges: Sequence[Tuple[int, int]],
    total_pixels: int,
) -> Dict[str, object]:
    target_low_bpp = _as_optional_float(codec_config, "target_low_bpp")
    target_low_bytes = _resolve_budget_target_bytes(total_pixels, target_low_bpp)
    if target_low_bytes is None:
        raise ValueError("Budget allocation requires target_low_bpp and total_pixels.")

    selected_modes = [str(section_analysis["minimum_mode"]) for section_analysis in section_analyses]
    fixed_bytes = _estimate_budget_fixed_bytes(global_keyframe, chunk_lengths, section_ranges, section_analyses)
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
    total_bytes = 0
    for key in ("global_keyframe",):
        value = codec_payload.get(key)
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()

    if codec_payload.get("section_payloads"):
        metadata_values: List[int] = [len(codec_payload.get("chunk_lengths", []))]
        metadata_values.extend(int(value) for value in codec_payload.get("chunk_lengths", []))
        for start, end in codec_payload.get("section_ranges", []):
            metadata_values.extend([int(start), int(end)])
        total_bytes += len(metadata_values) * 4

        for section_payload in codec_payload.get("section_payloads", []):
            total_bytes += 4  # mode id
            heuristic_scores = section_payload.get("heuristic_scores", {})
            total_bytes += len(heuristic_scores) * 4
            for anchor_block in section_payload.get("anchor_blocks", []):
                total_bytes += 8  # start + length
                anchor_payload = anchor_block.get("payload")
                if isinstance(anchor_payload, dict):
                    total_bytes += _estimate_refresh_anchor_bytes(anchor_payload)
        return int(total_bytes)

    for payload_key in ("section_anchor_payloads", "section_tail_payloads"):
        for payload in codec_payload.get(payload_key, []):
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

    metadata_values: List[int] = [len(codec_payload.get("chunk_lengths", []))]
    metadata_values.extend(int(value) for value in codec_payload.get("chunk_lengths", []))
    for start, end in codec_payload.get("section_ranges", []):
        metadata_values.extend([int(start), int(end)])
    total_bytes += len(metadata_values) * 4

    for payload_key in ("section_anchor_payloads", "section_tail_payloads"):
        for payload in codec_payload.get(payload_key, []):
            for shape_key in ("reduced_shape", "original_shape"):
                total_bytes += len(payload.get(shape_key, [])) * 4

    return int(total_bytes)


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

    keyframe_dtype = torch.float16 if _as_str(codec_config, "keyframe_dtype") == "float16" else torch.float32
    global_keyframe = clean_full_latents[:, :1].to(keyframe_dtype).contiguous()
    section_ranges = make_section_ranges(
        total_latent_frames=int(clean_full_latents.shape[1]),
        section_span_latents=_as_int(codec_config, "section_span_latents"),
        start_index=1,
    )

    target_low_bpp = _as_optional_float(codec_config, "target_low_bpp")
    decoded_history = global_keyframe.float()
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
            global_keyframe=global_keyframe,
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

    section_payloads: List[Dict[str, object]] = []
    cheap_tail_factor = (
        None
        if train_mixed_dual_tail_spatial_factor is None
        else int(train_mixed_dual_tail_spatial_factor)
    )
    cheap_tail_ratio = float(train_mixed_dual_tail_ratio)
    for section_analysis, selected_mode in zip(section_analyses, selected_modes):
        encoded_anchor_blocks = list(section_analysis["encoded_blocks_by_mode"][selected_mode])
        dual_tail_fidelity = None
        if selected_mode == DUAL_REFRESH_SECTION_MODE:
            dual_tail_fidelity = "full"
            if (
                cheap_tail_factor is not None
                and cheap_tail_factor > 1
                and cheap_tail_ratio > 0.0
                and len(encoded_anchor_blocks) > 1
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
        section_payloads.append(
            {
                "mode": str(selected_mode),
                "anchor_blocks": encoded_anchor_blocks,
                "heuristic_scores": dict(section_analysis["heuristic_scores"]),
                "dual_tail_fidelity": dual_tail_fidelity,
            }
        )

    return {
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "codec_config": dict(codec_config) if isinstance(codec_config, dict) else None,
        "global_keyframe": _maybe_to_cpu(global_keyframe, move_to_cpu),
        "section_ranges": [(int(start), int(end)) for start, end in section_ranges],
        "section_payloads": section_payloads,
        "mode_selection_strategy": mode_selection_strategy,
        "target_low_bpp": budget_result["target_low_bpp"],
        "target_low_bytes": budget_result["target_low_bytes"],
        "estimated_selected_bytes": budget_result["estimated_selected_bytes"],
        "estimated_fixed_bytes": budget_result["estimated_fixed_bytes"],
    }


def decode_anchor_plus_tail_latents(codec_payload: Dict[str, object], learned_tail_codec=None) -> torch.Tensor:
    global_keyframe = codec_payload["global_keyframe"].float()
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
            decoded_anchor_blocks: List[Tuple[int, torch.Tensor]] = []
            for anchor_block in section_payload.get("anchor_blocks", []):
                anchor_payload = anchor_block["payload"]
                decoded_anchor = decode_refresh_anchor(anchor_payload).to(device=target_device, dtype=torch.float32)
                decoded_anchor_blocks.append((int(anchor_block["start"]), decoded_anchor))
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
