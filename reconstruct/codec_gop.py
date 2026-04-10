import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from reconstruct.learned_codec import LEARNED_TAIL_CODEC_TYPE


EPS = 1e-8
TRILINEAR_TAIL_CODEC_TYPE = "trilinear"


def _get_config_value(codec_config, key: str):
    if isinstance(codec_config, dict):
        return codec_config[key]
    return getattr(codec_config, key)


def _as_int(codec_config, key: str) -> int:
    return int(_get_config_value(codec_config, key))


def _as_str(codec_config, key: str) -> str:
    return str(_get_config_value(codec_config, key))


def _tail_codec_type(codec_config) -> str:
    if isinstance(codec_config, dict):
        return str(codec_config.get("tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))
    return str(getattr(codec_config, "tail_codec_type", TRILINEAR_TAIL_CODEC_TYPE))


def _maybe_to_cpu(tensor: torch.Tensor, move_to_cpu: bool) -> torch.Tensor:
    if move_to_cpu:
        return tensor.cpu().contiguous()
    return tensor.contiguous()


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
    tail_codec_type = _tail_codec_type(codec_config)
    if tail_codec_type not in {TRILINEAR_TAIL_CODEC_TYPE, LEARNED_TAIL_CODEC_TYPE}:
        raise ValueError(
            f"Unsupported tail_codec_type={tail_codec_type}. "
            f"Expected {TRILINEAR_TAIL_CODEC_TYPE} or {LEARNED_TAIL_CODEC_TYPE}."
        )


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
    learned_tail_codec=None,
    use_ste_quant: bool = False,
    move_to_cpu: bool = True,
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

    section_anchor_payloads: List[Dict[str, object]] = []
    section_tail_payloads: List[Dict[str, object]] = []
    for section_start, section_end in section_ranges:
        section_latents = clean_full_latents[:, section_start:section_end]
        anchor_span = min(_as_int(codec_config, "anchor_span_latents"), section_latents.shape[1])
        section_anchor = section_latents[:, :anchor_span].contiguous()
        section_tail = section_latents[:, anchor_span:].contiguous()

        anchor_payload = encode_refresh_anchor(section_anchor, codec_config, move_to_cpu=move_to_cpu)
        decoded_anchor = decode_refresh_anchor(anchor_payload)
        tail_payload = encode_section_tail_residual(
            section_tail,
            decoded_anchor,
            codec_config,
            learned_tail_codec=learned_tail_codec,
            use_ste_quant=use_ste_quant,
            move_to_cpu=move_to_cpu,
        )
        section_anchor_payloads.append(anchor_payload)
        section_tail_payloads.append(tail_payload)

    return {
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "codec_config": dict(codec_config) if isinstance(codec_config, dict) else None,
        "global_keyframe": _maybe_to_cpu(global_keyframe, move_to_cpu),
        "section_ranges": [(int(start), int(end)) for start, end in section_ranges],
        "section_anchor_payloads": section_anchor_payloads,
        "section_tail_payloads": section_tail_payloads,
    }


def decode_anchor_plus_tail_latents(codec_payload: Dict[str, object], learned_tail_codec=None) -> torch.Tensor:
    global_keyframe = codec_payload["global_keyframe"].float()
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
        if decoded_anchor.device != decoded_tail.device:
            decoded_anchor = decoded_anchor.to(device=decoded_tail.device, dtype=torch.float32)
        sections.append(torch.cat([decoded_anchor, decoded_tail], dim=1))

    remainder = torch.cat(sections, dim=1)
    global_keyframe = global_keyframe.to(device=remainder.device, dtype=torch.float32)
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
