import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruct.decoder import (
    COMPARE_MODE_CHOICES,
    VAE_DECODE_MODE_CHOICES,
    CompareExportConfig,
    ModelBundle,
    build_compare_export_config,
    decode_payload_to_output_path,
    load_payload,
    resolve_device,
)
from reconstruct.external_entropy import (
    deserialize_external_entropy_payload,
    read_external_entropy_payload,
    serialize_external_entropy_payload,
)
from reconstruct.latent_io import (
    DEFAULT_BASE_MODEL_PATH,
    LATENT_FORMAT_V2,
    build_chunk_frame_ranges,
    compute_bpp_from_total_pixels,
    split_full_latents,
)
from reconstruct.recover import (
    DEFAULT_RECOVER_HISTORY_SOURCE,
    DEFAULT_RECOVER_INIT_MODE,
    DEFAULT_RECOVER_LOW_GUIDANCE_SCALE,
    DEFAULT_RECOVER_START_SIGMA,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_WEIGHT_DTYPE,
    CodecConfig,
    DistributedContext,
    PreparedSequence,
    SequenceMetadata,
    decode_low_latents,
    load_learned_tail_codec,
    load_recover_config,
    load_transformer_bundle,
    normalize_history_sizes,
    parse_weight_dtype,
    RECOVER_HISTORY_SOURCE_LOW,
    RECOVER_HISTORY_SOURCE_RECOVERED,
    RECOVER_INIT_MODE_LOW,
    RECOVER_INIT_MODE_LOW_NOISE,
    RECOVER_INIT_MODE_NOISE,
    reconstruct_sequence,
    resolve_checkpoint_dir_for_inference,
    resolve_infer_value,
)


ROOT_DIR = "reconstruct/outputs_CNN"
DEFAULT_INPUT_SUBDIR = "enc_latents"
DEFAULT_OUTPUT_SUBDIR = "Videos"
TRANSPORT_INPUT_MODE_CHOICES = ("auto", "low_latents", "simulate_entropy", "entropy_bin")


@dataclass
class RecoveryRuntime:
    checkpoint_dir: Path
    base_model_path: str
    codec_config: CodecConfig
    history_sizes: List[Optional[int]]
    latent_window_size: int
    anchor_span_latents: int
    weight_dtype: torch.dtype
    fix_anchor_during_denoise: bool
    learned_tail_codec: Optional[torch.nn.Module]
    transformer: torch.nn.Module
    scheduler: object
    prompt_embeds: torch.Tensor
    latent_channels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode transported low-rate payloads by recovering recover_latents and then decoding them into videos."
    )
    parser.add_argument("--root_dir", "-R", type=Path, default=Path(ROOT_DIR))
    parser.add_argument(
        "--input_dir",
        "-I",
        type=Path,
        default=None,
        help="Default: auto-detect <root_dir>/enc_latents first, then <root_dir>/low_latents.",
    )
    parser.add_argument(
        "--output_dir",
        "-O",
        type=Path,
        default=None,
        help=f"Default: <root_dir>/{DEFAULT_OUTPUT_SUBDIR}",
    )
    parser.add_argument(
        "--recover_latent_output_dir",
        type=Path,
        default=None,
        help="Optional directory to save the internally reconstructed recover_latents.",
    )
    parser.add_argument(
        "--metrics_dir",
        type=Path,
        default=None,
        help="Default: <root_dir>/metrics. Used to resolve checkpoint_dir when not passed explicitly.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=None,
        help="Recover checkpoint directory. If omitted, real_decoder will try metrics/*.json first.",
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
        type=int,
        default=None,
        help="If omitted, prefer metrics[].num_inference_steps and then fall back to the recover default.",
    )
    parser.add_argument("--source_video_dir", type=Path, default=Path("reconstruct/videos"))
    parser.add_argument("--compare_output_dir", type=Path, default=None)
    parser.add_argument("--compare_mode", type=str, default="both", choices=COMPARE_MODE_CHOICES)
    parser.add_argument(
        "--vae_decode_mode",
        type=str,
        default="auto",
        choices=VAE_DECODE_MODE_CHOICES,
        help="auto/full decodes recovered latent chunks as one VAE sequence to reduce chunk-boundary flicker.",
    )
    parser.add_argument(
        "--recover_overlap_latents",
        type=int,
        default=None,
        help="Number of latent timesteps overlapped between adjacent recover windows. If omitted, prefer metrics[].recover_overlap_latents.",
    )
    parser.add_argument(
        "--recover_history_source",
        type=str,
        default=None,
        choices=[RECOVER_HISTORY_SOURCE_LOW, RECOVER_HISTORY_SOURCE_RECOVERED],
        help="If omitted, prefer metrics[].recover_history_source and then fall back to the recover default.",
    )
    parser.add_argument(
        "--recover_init_mode",
        type=str,
        default=None,
        choices=[RECOVER_INIT_MODE_NOISE, RECOVER_INIT_MODE_LOW_NOISE, RECOVER_INIT_MODE_LOW],
        help="If omitted, prefer metrics[].recover_init_mode and then fall back to the recover default.",
    )
    parser.add_argument(
        "--recover_start_sigma",
        type=float,
        default=None,
        help="If omitted, prefer metrics[].recover_start_sigma and then fall back to the recover default.",
    )
    parser.add_argument(
        "--recover_low_guidance_scale",
        type=float,
        default=None,
        help="If omitted, prefer metrics[].recover_low_guidance_scale and then fall back to the recover default.",
    )
    parser.add_argument(
        "--transport_input_mode",
        type=str,
        default="auto",
        choices=TRANSPORT_INPUT_MODE_CHOICES,
        help="How low_latent payloads should be treated at the receiver. 'auto' prefers real .bin inputs and otherwise simulates entropy transport in memory.",
    )
    return parser.parse_args()


def discover_input_paths(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    input_paths = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in {".bin", ".pt"}
    ]
    if not input_paths:
        raise FileNotFoundError(
            f"No supported payload files (.bin or .pt) were found in: {input_dir}"
        )
    return input_paths


def is_low_latent_payload(payload: Dict[str, object]) -> bool:
    format_version = str(payload.get("format_version", ""))
    if format_version.startswith("helios_low_latent_"):
        return True
    return (
        ("section_anchor_payloads" in payload and "section_tail_payloads" in payload)
        or ("quantized_remainder" in payload and "keyframe" in payload)
    )


def infer_low_latent_channels(payload: Dict[str, object], latent_path: Path) -> int:
    if "global_keyframe" in payload:
        global_keyframe = payload["global_keyframe"]
        if isinstance(global_keyframe, torch.Tensor):
            return int(global_keyframe.shape[0])
        if isinstance(global_keyframe, dict):
            if "original_shape" in global_keyframe and global_keyframe["original_shape"]:
                return int(global_keyframe["original_shape"][0])
            if "reduced_shape" in global_keyframe and global_keyframe["reduced_shape"]:
                return int(global_keyframe["reduced_shape"][0])
    if "keyframe" in payload:
        return int(payload["keyframe"].shape[0])
    raise KeyError(
        f"Low latent payload {latent_path} is missing both global_keyframe and keyframe, "
        "so latent_channels cannot be inferred."
    )


def load_metrics_payload(metrics_path: Path) -> Optional[Dict[str, object]]:
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_metrics_path(metrics_dir: Path, relative_path: Path) -> Path:
    return metrics_dir / relative_path.with_suffix(".json")


def load_entropy_metrics_payload(entropy_metrics_path: Path) -> Optional[Dict[str, object]]:
    if not entropy_metrics_path.exists():
        return None
    with entropy_metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_entropy_metrics_path(entropy_metrics_dir: Path, relative_path: Path) -> Path:
    return entropy_metrics_dir / relative_path.with_suffix(".json")


def resolve_default_input_dir(root_dir: Path) -> Path:
    for subdir_name in ("enc_latents", "low_latents"):
        candidate = (root_dir / subdir_name).resolve()
        if candidate.exists():
            return candidate
    return (root_dir / DEFAULT_INPUT_SUBDIR).resolve()


def resolve_transport_input_mode(
    cli_mode: str,
    latent_path: Path,
    *,
    is_low_payload: bool,
) -> str:
    suffix = latent_path.suffix.lower()
    if suffix == ".bin":
        if cli_mode not in {"auto", "entropy_bin"}:
            raise ValueError(
                f"Transport mode {cli_mode!r} is incompatible with entropy bitstream input {latent_path}."
            )
        return "entropy_bin"

    if is_low_payload:
        if cli_mode == "low_latents":
            return "low_latents"
        if cli_mode in {"auto", "simulate_entropy"}:
            return "simulate_entropy"
        raise ValueError(
            f"Transport mode {cli_mode!r} is incompatible with low_latent payload input {latent_path}."
        )

    if cli_mode != "auto":
        raise ValueError(
            f"Transport mode {cli_mode!r} only supports low_latent or entropy bitstream inputs, got {latent_path}."
        )
    return "direct_decode"


def build_simulated_entropy_metrics_payload(
    low_payload: Dict[str, object],
    relative_path: Path,
    latent_path: Path,
    entropy_metrics_dir: Path,
    entropy_summary: Dict[str, object],
) -> Dict[str, object]:
    entropy_metrics_path = resolve_entropy_metrics_path(
        entropy_metrics_dir=entropy_metrics_dir,
        relative_path=relative_path,
    )
    metrics_payload = load_entropy_metrics_payload(entropy_metrics_path) or {}
    total_pixels = (
        int(low_payload["source_num_frames"])
        * int(low_payload["source_width"])
        * int(low_payload["source_height"])
    )
    entropy_codec_bytes = int(metrics_payload.get("entropy_codec_bytes", entropy_summary["entropy_codec_bytes"]))
    return {
        **metrics_payload,
        "input_low_latent_path": str(metrics_payload.get("input_low_latent_path") or latent_path),
        "enc_latent_path": metrics_payload.get("enc_latent_path"),
        "transport_storage": str(metrics_payload.get("transport_storage", "memory_only")),
        "entropy_codec_bytes": entropy_codec_bytes,
        "entropy_bpp": float(
            metrics_payload.get(
                "entropy_bpp",
                compute_bpp_from_total_pixels(entropy_codec_bytes, total_pixels),
            )
        ),
        "entropy_codec": str(metrics_payload.get("entropy_codec", entropy_summary["entropy_codec"])),
        "field_strategy": str(metrics_payload.get("field_strategy", entropy_summary["field_strategy"])),
        "block_lengths": metrics_payload.get("block_lengths", entropy_summary["block_lengths"]),
        "section_anchor_entropy_bytes": metrics_payload.get(
            "section_anchor_entropy_bytes",
            entropy_summary.get("section_anchor_entropy_bytes", []),
        ),
        "section_tail_entropy_bytes": metrics_payload.get(
            "section_tail_entropy_bytes",
            entropy_summary.get("section_tail_entropy_bytes", []),
        ),
        "section_entropy_bytes": metrics_payload.get(
            "section_entropy_bytes",
            entropy_summary.get("section_entropy_bytes", []),
        ),
        "section_entropy_bpp": metrics_payload.get(
            "section_entropy_bpp",
            [
                float(section_bytes * 8.0 / float(total_pixels))
                for section_bytes in entropy_summary.get("section_entropy_bytes", [])
            ],
        ),
    }


def resolve_requested_checkpoint_dir(
    cli_checkpoint_dir: Optional[Path],
    metrics_payload: Optional[Dict[str, object]],
    root_dir: Path,
    latent_path: Path,
) -> Path:
    if cli_checkpoint_dir is not None:
        return cli_checkpoint_dir.resolve()

    if metrics_payload is not None and metrics_payload.get("checkpoint_dir"):
        return Path(str(metrics_payload["checkpoint_dir"])).resolve()

    fallback_candidates = [
        (root_dir / "checkpoints").resolve(),
        (root_dir.parent / "checkpoints").resolve(),
    ]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Unable to resolve checkpoint_dir for {latent_path}. "
        "Please pass --checkpoint_dir explicitly or keep metrics/*.json alongside the transported payloads."
    )


def resolve_num_inference_steps(cli_value: Optional[int], metrics_payload: Optional[Dict[str, object]]) -> int:
    if cli_value is not None:
        if cli_value <= 0:
            raise ValueError(f"--num_inference_steps must be > 0, got {cli_value}.")
        return int(cli_value)
    if metrics_payload is not None and metrics_payload.get("num_inference_steps") is not None:
        return int(metrics_payload["num_inference_steps"])
    return DEFAULT_NUM_INFERENCE_STEPS


def resolve_recover_overlap_latents(cli_value: Optional[int], metrics_payload: Optional[Dict[str, object]]) -> int:
    if cli_value is not None:
        if cli_value < 0:
            raise ValueError(f"--recover_overlap_latents must be >= 0, got {cli_value}.")
        return int(cli_value)
    if metrics_payload is not None and metrics_payload.get("recover_overlap_latents") is not None:
        metrics_value = int(metrics_payload["recover_overlap_latents"])
        if metrics_value < 0:
            raise ValueError(f"metrics recover_overlap_latents must be >= 0, got {metrics_value}.")
        return metrics_value
    return 0


def resolve_recover_history_source(cli_value: Optional[str], metrics_payload: Optional[Dict[str, object]]) -> str:
    value = cli_value
    if value is None and metrics_payload is not None:
        value = metrics_payload.get("recover_history_source")
    if value is None:
        value = DEFAULT_RECOVER_HISTORY_SOURCE
    value = str(value)
    if value not in {RECOVER_HISTORY_SOURCE_LOW, RECOVER_HISTORY_SOURCE_RECOVERED}:
        raise ValueError(f"Unsupported recover_history_source={value}.")
    return value


def resolve_recover_init_mode(cli_value: Optional[str], metrics_payload: Optional[Dict[str, object]]) -> str:
    value = cli_value
    if value is None and metrics_payload is not None:
        value = metrics_payload.get("recover_init_mode")
    if value is None:
        value = DEFAULT_RECOVER_INIT_MODE
    value = str(value)
    if value not in {RECOVER_INIT_MODE_NOISE, RECOVER_INIT_MODE_LOW_NOISE, RECOVER_INIT_MODE_LOW}:
        raise ValueError(f"Unsupported recover_init_mode={value}.")
    return value


def resolve_recover_start_sigma(
    cli_value: Optional[float],
    metrics_payload: Optional[Dict[str, object]],
) -> Optional[float]:
    value = cli_value
    if value is None and metrics_payload is not None:
        value = metrics_payload.get("recover_start_sigma", DEFAULT_RECOVER_START_SIGMA)
    if value is None:
        return DEFAULT_RECOVER_START_SIGMA
    value = float(value)
    if value <= 0.0 or value > 0.999:
        raise ValueError(f"recover_start_sigma must satisfy 0 < sigma <= 0.999, got {value}.")
    return value


def resolve_recover_low_guidance_scale(
    cli_value: Optional[float],
    metrics_payload: Optional[Dict[str, object]],
) -> float:
    value = cli_value
    if value is None and metrics_payload is not None:
        value = metrics_payload.get("recover_low_guidance_scale", DEFAULT_RECOVER_LOW_GUIDANCE_SCALE)
    if value is None:
        value = DEFAULT_RECOVER_LOW_GUIDANCE_SCALE
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"recover_low_guidance_scale must be in [0, 1], got {value}.")
    return value


def validate_low_payload_compatibility(
    low_payload: Dict[str, object],
    runtime: RecoveryRuntime,
    latent_path: Path,
) -> None:
    payload_codec = dict(low_payload.get("codec_config") or low_payload.get("codec_config_v2") or {})
    if not payload_codec:
        return

    runtime_codec = {
        "section_span_latents": int(runtime.codec_config.section_span_latents),
        "anchor_span_latents": int(runtime.codec_config.anchor_span_latents),
        "tail_span_latents": int(runtime.codec_config.tail_span_latents),
        "temporal_factor": int(runtime.codec_config.temporal_factor),
        "spatial_factor": int(runtime.codec_config.spatial_factor),
        "keyframe_codec_mode": str(runtime.codec_config.keyframe_codec_mode),
        "keyframe_quant_dtype": str(runtime.codec_config.keyframe_quant_dtype),
        "keyframe_spatial_factor": int(runtime.codec_config.keyframe_spatial_factor),
        "anchor_quant_dtype": str(runtime.codec_config.anchor_quant_dtype),
        "anchor_spatial_factor": int(runtime.codec_config.anchor_spatial_factor),
        "tail_codec_type": str(runtime.codec_config.tail_codec_type),
    }
    mismatches = []
    for key, runtime_value in runtime_codec.items():
        if key not in payload_codec:
            continue
        payload_value = payload_codec[key]
        if str(payload_value) != str(runtime_value):
            mismatches.append(f"{key}: payload={payload_value} checkpoint={runtime_value}")

    if mismatches:
        raise ValueError(
            f"Low latent payload {latent_path} is incompatible with checkpoint {runtime.checkpoint_dir}: "
            + "; ".join(mismatches)
        )


def build_sequence_from_low_payload(
    latent_path: Path,
    low_payload: Dict[str, object],
    low_full_latents: torch.Tensor,
    metrics_payload: Optional[Dict[str, object]],
) -> PreparedSequence:
    if "chunk_lengths" not in low_payload:
        raise KeyError(f"Low latent payload {latent_path} is missing chunk_lengths.")

    chunk_lengths = [int(value) for value in low_payload["chunk_lengths"]]
    chunk_frame_ranges = build_chunk_frame_ranges([4 * (length - 1) + 1 for length in chunk_lengths])
    source_num_frames = int(low_payload["source_num_frames"])
    source_width = int(low_payload["source_width"])
    source_height = int(low_payload["source_height"])
    total_pixels = source_num_frames * source_width * source_height

    raw_file_bytes = latent_path.stat().st_size
    raw_bpp = float(low_payload.get("low_bpp", compute_bpp_from_total_pixels(raw_file_bytes, total_pixels)))
    if metrics_payload is not None:
        raw_file_bytes = int(metrics_payload.get("raw_file_bytes", raw_file_bytes))
        raw_bpp = float(metrics_payload.get("raw_bpp", raw_bpp))

    metadata = SequenceMetadata(
        source_fps=float(low_payload["source_fps"]),
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
        path=latent_path.resolve(),
        metadata=metadata,
        clean_full_latents=low_full_latents,
        low_codec_payload=low_payload,
        low_full_latents=low_full_latents,
    )


def build_recover_payload_from_low(
    latent_path: Path,
    low_payload: Dict[str, object],
    recovered_full_latents: torch.Tensor,
    runtime: RecoveryRuntime,
    metrics_payload: Optional[Dict[str, object]],
    transport_mode: str,
    transport_file_bytes: int,
    transport_bpp: float,
    entropy_metrics_payload: Optional[Dict[str, object]],
    recover_overlap_latents: int,
    recover_history_source: str,
    recover_init_mode: str,
    recover_start_sigma: Optional[float],
    recover_low_guidance_scale: float,
) -> Dict[str, object]:
    chunk_lengths = [int(value) for value in low_payload["chunk_lengths"]]
    restored_chunks = split_full_latents(recovered_full_latents, chunk_lengths)
    chunk_frame_ranges = build_chunk_frame_ranges([4 * (length - 1) + 1 for length in chunk_lengths])

    recovery_metadata = {
        "input_path": str(low_payload.get("input_path", latent_path)),
        "transport_path": str(latent_path),
        "transport_mode": str(transport_mode),
        "transport_file_bytes": int(transport_file_bytes),
        "transport_bpp": float(transport_bpp),
        "checkpoint_dir": str(runtime.checkpoint_dir),
        "recover_overlap_latents": int(recover_overlap_latents),
        "recover_history_source": str(recover_history_source),
        "recover_init_mode": str(recover_init_mode),
        "recover_start_sigma": recover_start_sigma,
        "recover_low_guidance_scale": float(recover_low_guidance_scale),
        "low_codec_bytes": int(low_payload.get("low_codec_bytes", latent_path.stat().st_size)),
        "low_bpp": float(low_payload.get("low_bpp", 0.0)),
        "section_ranges": [
            [int(value) for value in section_range]
            for section_range in low_payload.get("section_ranges", [])
        ],
        "dynamic_rate_policy": low_payload.get("dynamic_rate_policy"),
        "section_motion_scores": list(low_payload.get("section_motion_scores", [])),
        "section_anchor_profiles": list(low_payload.get("section_anchor_profiles", [])),
        "section_tail_quality_scales": list(low_payload.get("section_tail_quality_scales", [])),
        "section_anchor_bytes": list(low_payload.get("section_anchor_bytes", [])),
        "section_tail_bytes": list(low_payload.get("section_tail_bytes", [])),
        "section_total_bytes": list(low_payload.get("section_total_bytes", [])),
        "section_bpp": list(low_payload.get("section_bpp", [])),
        "video_mean_motion_score": float(low_payload.get("video_mean_motion_score", 0.0)),
        "video_mean_tail_quality_scale": float(low_payload.get("video_mean_tail_quality_scale", 1.0)),
        "video_section_bpp_mean": float(low_payload.get("video_section_bpp_mean", 0.0)),
        "generated_by": "reconstruct/real_decoder.py",
    }
    if transport_mode == "low_latents":
        recovery_metadata["low_latent_path"] = str(latent_path)
    elif transport_mode == "simulated_entropy":
        recovery_metadata["low_latent_path"] = str(latent_path)
        recovery_metadata["simulated_entropy_transport"] = True
    elif transport_mode == "entropy_bin":
        recovery_metadata["entropy_bin_path"] = str(latent_path)
    if metrics_payload is not None and metrics_payload.get("low_latent_path") is not None:
        recovery_metadata["low_latent_path"] = str(metrics_payload["low_latent_path"])
    if entropy_metrics_payload is not None:
        if entropy_metrics_payload.get("enc_latent_path") is not None:
            recovery_metadata["entropy_bin_path"] = str(entropy_metrics_payload["enc_latent_path"])
        if entropy_metrics_payload.get("entropy_codec_bytes") is not None:
            recovery_metadata["entropy_codec_bytes"] = int(entropy_metrics_payload["entropy_codec_bytes"])
        if entropy_metrics_payload.get("entropy_bpp") is not None:
            recovery_metadata["entropy_bpp"] = float(entropy_metrics_payload["entropy_bpp"])
        if entropy_metrics_payload.get("entropy_codec") is not None:
            recovery_metadata["entropy_codec"] = str(entropy_metrics_payload["entropy_codec"])
        if entropy_metrics_payload.get("transport_storage") is not None:
            recovery_metadata["transport_storage"] = str(entropy_metrics_payload["transport_storage"])
        if entropy_metrics_payload.get("field_strategy") is not None:
            recovery_metadata["field_strategy"] = str(entropy_metrics_payload["field_strategy"])
        for key in (
            "section_anchor_entropy_bytes",
            "section_tail_entropy_bytes",
            "section_entropy_bytes",
            "section_entropy_bpp",
        ):
            if entropy_metrics_payload.get(key) is not None:
                recovery_metadata[key] = entropy_metrics_payload[key]
    if metrics_payload is not None and metrics_payload.get("raw_bpp") is not None:
        recovery_metadata["raw_bpp"] = float(metrics_payload["raw_bpp"])

    return {
        "format_version": LATENT_FORMAT_V2,
        "source_fps": float(low_payload["source_fps"]),
        "source_num_frames": int(low_payload["source_num_frames"]),
        "source_width": int(low_payload["source_width"]),
        "source_height": int(low_payload["source_height"]),
        "chunk_frame_ranges": [list(frame_range) for frame_range in chunk_frame_ranges],
        "latent_chunks": restored_chunks,
        "model_path": str(runtime.base_model_path),
        "recovery_metadata": recovery_metadata,
    }


def maybe_save_recover_payload(
    recover_payload: Dict[str, object],
    relative_path: Path,
    recover_latent_output_dir: Optional[Path],
) -> Optional[Path]:
    if recover_latent_output_dir is None:
        return None
    recover_latent_path = (recover_latent_output_dir / relative_path).with_suffix(".pt")
    recover_latent_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recover_payload, recover_latent_path)
    return recover_latent_path


def load_recovery_runtime(
    requested_checkpoint_dir: Path,
    latent_channels: int,
    cli_base_model_path: str,
    cli_weight_dtype: str,
    device: torch.device,
    runtime_cache: Dict[Path, RecoveryRuntime],
) -> RecoveryRuntime:
    checkpoint_dir = resolve_checkpoint_dir_for_inference(requested_checkpoint_dir)
    cached_runtime = runtime_cache.get(checkpoint_dir)
    if cached_runtime is not None:
        if cached_runtime.latent_channels != latent_channels:
            raise ValueError(
                f"Checkpoint {checkpoint_dir} was already loaded for latent_channels={cached_runtime.latent_channels}, "
                f"but the current low payload requires latent_channels={latent_channels}."
            )
        return cached_runtime

    try:
        checkpoint_config = load_recover_config(checkpoint_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load recover checkpoint config from {checkpoint_dir}. "
            "real_decoder reuses reconstruct/recover.py's checkpoint contract, "
            "so the checkpoint format and codec config must be supported by the current repo."
        ) from exc
    base_model_path = resolve_infer_value(
        cli_value=cli_base_model_path,
        default_value=DEFAULT_BASE_MODEL_PATH,
        checkpoint_value=checkpoint_config.get("base_model_path"),
    )
    weight_dtype_name = resolve_infer_value(
        cli_value=cli_weight_dtype,
        default_value=DEFAULT_WEIGHT_DTYPE,
        checkpoint_value=checkpoint_config.get("weight_dtype"),
    )
    codec_config = CodecConfig(**checkpoint_config["codec_config"])
    history_sizes = normalize_history_sizes(checkpoint_config["history_sizes"])
    latent_window_size = int(checkpoint_config.get("section_span_latents", checkpoint_config["latent_window_size"]))
    anchor_span_latents = int(checkpoint_config.get("anchor_span_latents", codec_config.anchor_span_latents))
    fix_anchor_during_denoise = bool(checkpoint_config.get("fix_anchor_during_denoise", True))
    weight_dtype = parse_weight_dtype(str(weight_dtype_name))

    learned_tail_codec = load_learned_tail_codec(
        checkpoint_dir=checkpoint_dir,
        codec_config=codec_config,
        latent_channels=latent_channels,
        device=device,
    )
    transformer, scheduler, prompt_embeds = load_transformer_bundle(
        base_model_path=str(base_model_path),
        device=device,
        weight_dtype=weight_dtype,
        checkpoint_dir=checkpoint_dir,
        gradient_checkpointing=False,
    )
    transformer.eval()

    runtime = RecoveryRuntime(
        checkpoint_dir=checkpoint_dir,
        base_model_path=str(base_model_path),
        codec_config=codec_config,
        history_sizes=history_sizes,
        latent_window_size=latent_window_size,
        anchor_span_latents=anchor_span_latents,
        weight_dtype=weight_dtype,
        fix_anchor_during_denoise=fix_anchor_during_denoise,
        learned_tail_codec=learned_tail_codec,
        transformer=transformer,
        scheduler=scheduler,
        prompt_embeds=prompt_embeds,
        latent_channels=latent_channels,
    )
    runtime_cache[checkpoint_dir] = runtime
    return runtime


def decode_low_latent_file_to_output_path(
    low_payload: Dict[str, object],
    latent_path: Path,
    relative_path: Path,
    output_path: Path,
    root_dir: Path,
    metrics_dir: Path,
    recover_latent_output_dir: Optional[Path],
    cli_checkpoint_dir: Optional[Path],
    cli_base_model_path: str,
    cli_weight_dtype: str,
    cli_num_inference_steps: Optional[int],
    seed: int,
    runtime_cache: Dict[Path, RecoveryRuntime],
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
    compare_config: Optional[CompareExportConfig],
    transport_mode: str,
    entropy_metrics_payload: Optional[Dict[str, object]],
    vae_decode_mode: str,
    cli_recover_overlap_latents: Optional[int],
    cli_recover_history_source: Optional[str],
    cli_recover_init_mode: Optional[str],
    cli_recover_start_sigma: Optional[float],
    cli_recover_low_guidance_scale: Optional[float],
) -> Path:
    metrics_path = resolve_metrics_path(metrics_dir=metrics_dir, relative_path=relative_path)
    metrics_payload = load_metrics_payload(metrics_path)
    recover_overlap_latents = resolve_recover_overlap_latents(cli_recover_overlap_latents, metrics_payload)
    recover_history_source = resolve_recover_history_source(cli_recover_history_source, metrics_payload)
    recover_init_mode = resolve_recover_init_mode(cli_recover_init_mode, metrics_payload)
    recover_start_sigma = resolve_recover_start_sigma(cli_recover_start_sigma, metrics_payload)
    recover_low_guidance_scale = resolve_recover_low_guidance_scale(
        cli_recover_low_guidance_scale,
        metrics_payload,
    )
    requested_checkpoint_dir = resolve_requested_checkpoint_dir(
        cli_checkpoint_dir=cli_checkpoint_dir,
        metrics_payload=metrics_payload,
        root_dir=root_dir,
        latent_path=latent_path,
    )
    latent_channels = infer_low_latent_channels(low_payload, latent_path)
    runtime = load_recovery_runtime(
        requested_checkpoint_dir=requested_checkpoint_dir,
        latent_channels=latent_channels,
        cli_base_model_path=cli_base_model_path,
        cli_weight_dtype=cli_weight_dtype,
        device=device,
        runtime_cache=runtime_cache,
    )
    validate_low_payload_compatibility(low_payload=low_payload, runtime=runtime, latent_path=latent_path)

    total_pixels = (
        int(low_payload["source_num_frames"])
        * int(low_payload["source_width"])
        * int(low_payload["source_height"])
    )
    transport_file_bytes = int(latent_path.stat().st_size)
    transport_bpp = compute_bpp_from_total_pixels(transport_file_bytes, total_pixels)
    if transport_mode in {"entropy_bin", "simulated_entropy"} and entropy_metrics_payload is not None:
        if entropy_metrics_payload.get("entropy_codec_bytes") is not None:
            transport_file_bytes = int(entropy_metrics_payload["entropy_codec_bytes"])
        if entropy_metrics_payload.get("entropy_bpp") is not None:
            transport_bpp = float(entropy_metrics_payload["entropy_bpp"])

    with torch.inference_mode():
        low_full_latents = decode_low_latents(
            codec_payload=low_payload,
            learned_tail_codec=runtime.learned_tail_codec,
        ).float().cpu().contiguous()

    sequence = build_sequence_from_low_payload(
        latent_path=latent_path,
        low_payload=low_payload,
        low_full_latents=low_full_latents,
        metrics_payload=metrics_payload,
    )
    recovered_full_latents = reconstruct_sequence(
        sequence=sequence,
        transformer=runtime.transformer,
        scheduler=runtime.scheduler,
        prompt_embeds=runtime.prompt_embeds,
        device=device,
        weight_dtype=runtime.weight_dtype,
        history_sizes=runtime.history_sizes,
        latent_window_size=runtime.latent_window_size,
        anchor_span_latents=runtime.anchor_span_latents,
        num_inference_steps=resolve_num_inference_steps(cli_num_inference_steps, metrics_payload),
        seed=seed,
        fix_anchor_during_denoise=runtime.fix_anchor_during_denoise,
        recover_overlap_latents=recover_overlap_latents,
        distributed_context=DistributedContext(is_distributed=False),
        recover_history_source=recover_history_source,
        recover_init_mode=recover_init_mode,
        recover_start_sigma=recover_start_sigma,
        recover_low_guidance_scale=recover_low_guidance_scale,
    )
    recover_payload = build_recover_payload_from_low(
        latent_path=latent_path,
        low_payload=low_payload,
        recovered_full_latents=recovered_full_latents,
        runtime=runtime,
        metrics_payload=metrics_payload,
        transport_mode=transport_mode,
        transport_file_bytes=transport_file_bytes,
        transport_bpp=transport_bpp,
        entropy_metrics_payload=entropy_metrics_payload,
        recover_overlap_latents=recover_overlap_latents,
        recover_history_source=recover_history_source,
        recover_init_mode=recover_init_mode,
        recover_start_sigma=recover_start_sigma,
        recover_low_guidance_scale=recover_low_guidance_scale,
    )
    saved_recover_path = maybe_save_recover_payload(
        recover_payload=recover_payload,
        relative_path=relative_path,
        recover_latent_output_dir=recover_latent_output_dir,
    )
    decoded_output_path = decode_payload_to_output_path(
        payload=recover_payload,
        latent_path=latent_path,
        relative_path=relative_path,
        output_path=output_path,
        cli_model_path=runtime.base_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
        compare_config=compare_config,
        vae_decode_mode=vae_decode_mode,
    )
    if saved_recover_path is not None:
        print(
            f"[real_decoder] saved_recover_latents={saved_recover_path} "
            f"from_transport={latent_path}"
        )
    return decoded_output_path


def decode_latent_file(
    latent_path: Path,
    input_dir: Path,
    output_dir: Path,
    root_dir: Path,
    metrics_dir: Path,
    recover_latent_output_dir: Optional[Path],
    cli_checkpoint_dir: Optional[Path],
    cli_base_model_path: str,
    cli_weight_dtype: str,
    cli_num_inference_steps: Optional[int],
    seed: int,
    runtime_cache: Dict[Path, RecoveryRuntime],
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
    compare_config: Optional[CompareExportConfig],
    transport_input_mode: str,
    vae_decode_mode: str,
    cli_recover_overlap_latents: Optional[int],
    cli_recover_history_source: Optional[str],
    cli_recover_init_mode: Optional[str],
    cli_recover_start_sigma: Optional[float],
    cli_recover_low_guidance_scale: Optional[float],
) -> Path:
    relative_path = latent_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".mp4")
    entropy_metrics_dir = (root_dir / "entropy_metrics").resolve()

    if latent_path.suffix.lower() == ".bin":
        resolve_transport_input_mode(
            transport_input_mode,
            latent_path,
            is_low_payload=False,
        )
        print(f"[real_decoder] input_mode=entropy_bin path={latent_path}")
        entropy_metrics_path = resolve_entropy_metrics_path(
            entropy_metrics_dir=entropy_metrics_dir,
            relative_path=relative_path,
        )
        entropy_metrics_payload = load_entropy_metrics_payload(entropy_metrics_path)
        payload = read_external_entropy_payload(latent_path)
        return decode_low_latent_file_to_output_path(
            low_payload=payload,
            latent_path=latent_path,
            relative_path=relative_path,
            output_path=output_path,
            root_dir=root_dir,
            metrics_dir=metrics_dir,
            recover_latent_output_dir=recover_latent_output_dir,
            cli_checkpoint_dir=cli_checkpoint_dir,
            cli_base_model_path=cli_base_model_path,
            cli_weight_dtype=cli_weight_dtype,
            cli_num_inference_steps=cli_num_inference_steps,
            seed=seed,
            runtime_cache=runtime_cache,
            model_bundle_cache=model_bundle_cache,
            device=device,
            compare_config=compare_config,
            transport_mode="entropy_bin",
            entropy_metrics_payload=entropy_metrics_payload,
            vae_decode_mode=vae_decode_mode,
            cli_recover_overlap_latents=cli_recover_overlap_latents,
            cli_recover_history_source=cli_recover_history_source,
            cli_recover_init_mode=cli_recover_init_mode,
            cli_recover_start_sigma=cli_recover_start_sigma,
            cli_recover_low_guidance_scale=cli_recover_low_guidance_scale,
        )

    payload = load_payload(latent_path)
    low_payload_input = is_low_latent_payload(payload)
    resolved_transport_mode = resolve_transport_input_mode(
        transport_input_mode,
        latent_path,
        is_low_payload=low_payload_input,
    )

    if low_payload_input and resolved_transport_mode == "simulate_entropy":
        print(f"[real_decoder] input_mode=simulate_entropy_from_low_latents path={latent_path}")
        transport_bytes, entropy_summary = serialize_external_entropy_payload(
            payload,
            relative_path=str(relative_path),
        )
        simulated_payload = deserialize_external_entropy_payload(
            transport_bytes,
            source=f"in_memory:{latent_path}",
        )
        entropy_metrics_payload = build_simulated_entropy_metrics_payload(
            low_payload=payload,
            relative_path=relative_path,
            latent_path=latent_path,
            entropy_metrics_dir=entropy_metrics_dir,
            entropy_summary=entropy_summary,
        )
        return decode_low_latent_file_to_output_path(
            low_payload=simulated_payload,
            latent_path=latent_path,
            relative_path=relative_path,
            output_path=output_path,
            root_dir=root_dir,
            metrics_dir=metrics_dir,
            recover_latent_output_dir=recover_latent_output_dir,
            cli_checkpoint_dir=cli_checkpoint_dir,
            cli_base_model_path=cli_base_model_path,
            cli_weight_dtype=cli_weight_dtype,
            cli_num_inference_steps=cli_num_inference_steps,
            seed=seed,
            runtime_cache=runtime_cache,
            model_bundle_cache=model_bundle_cache,
            device=device,
            compare_config=compare_config,
            transport_mode="simulated_entropy",
            entropy_metrics_payload=entropy_metrics_payload,
            vae_decode_mode=vae_decode_mode,
            cli_recover_overlap_latents=cli_recover_overlap_latents,
            cli_recover_history_source=cli_recover_history_source,
            cli_recover_init_mode=cli_recover_init_mode,
            cli_recover_start_sigma=cli_recover_start_sigma,
            cli_recover_low_guidance_scale=cli_recover_low_guidance_scale,
        )

    if low_payload_input:
        print(f"[real_decoder] input_mode=low_latents path={latent_path}")
        return decode_low_latent_file_to_output_path(
            low_payload=payload,
            latent_path=latent_path,
            relative_path=relative_path,
            output_path=output_path,
            root_dir=root_dir,
            metrics_dir=metrics_dir,
            recover_latent_output_dir=recover_latent_output_dir,
            cli_checkpoint_dir=cli_checkpoint_dir,
            cli_base_model_path=cli_base_model_path,
            cli_weight_dtype=cli_weight_dtype,
            cli_num_inference_steps=cli_num_inference_steps,
            seed=seed,
            runtime_cache=runtime_cache,
            model_bundle_cache=model_bundle_cache,
            device=device,
            compare_config=compare_config,
            transport_mode=resolved_transport_mode,
            entropy_metrics_payload=None,
            vae_decode_mode=vae_decode_mode,
            cli_recover_overlap_latents=cli_recover_overlap_latents,
            cli_recover_history_source=cli_recover_history_source,
            cli_recover_init_mode=cli_recover_init_mode,
            cli_recover_start_sigma=cli_recover_start_sigma,
            cli_recover_low_guidance_scale=cli_recover_low_guidance_scale,
        )

    print(f"[real_decoder] input_mode=direct_decode path={latent_path}")
    return decode_payload_to_output_path(
        payload=payload,
        latent_path=latent_path,
        relative_path=relative_path,
        output_path=output_path,
        cli_model_path=cli_base_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
        compare_config=compare_config,
        vae_decode_mode=vae_decode_mode,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    root_dir = args.root_dir.resolve()
    input_dir = args.input_dir.resolve() if args.input_dir else resolve_default_input_dir(root_dir)
    output_dir = args.output_dir.resolve() if args.output_dir else (root_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    metrics_dir = args.metrics_dir.resolve() if args.metrics_dir else (root_dir / "metrics").resolve()
    recover_latent_output_dir = (
        args.recover_latent_output_dir.resolve() if args.recover_latent_output_dir else None
    )
    compare_config = build_compare_export_config(args, root_dir)
    latent_paths = discover_input_paths(input_dir)

    print(f"[real_decoder] input_dir={input_dir}")
    print(f"[real_decoder] output_dir={output_dir}")
    print(f"[real_decoder] metrics_dir={metrics_dir}")
    print(f"[real_decoder] checkpoint_dir={args.checkpoint_dir.resolve() if args.checkpoint_dir else 'auto'}")
    print(f"[real_decoder] base_model_path={args.base_model_path or 'from-checkpoint-or-latent-metadata'}")
    print(f"[real_decoder] recover_latent_output_dir={recover_latent_output_dir or 'off'}")
    print(f"[real_decoder] transport_input_mode={args.transport_input_mode}")
    print(f"[real_decoder] vae_decode_mode={args.vae_decode_mode}")
    print(
        f"[real_decoder] recover_overlap_latents="
        f"{args.recover_overlap_latents if args.recover_overlap_latents is not None else 'auto'}"
    )
    print(
        f"[real_decoder] recover_stabilization="
        f"history_source={args.recover_history_source if args.recover_history_source is not None else 'auto'} "
        f"init_mode={args.recover_init_mode if args.recover_init_mode is not None else 'auto'} "
        f"start_sigma={args.recover_start_sigma if args.recover_start_sigma is not None else 'auto'} "
        f"low_guidance={args.recover_low_guidance_scale if args.recover_low_guidance_scale is not None else 'auto'}"
    )
    if compare_config is None:
        print("[real_decoder] compare_mode=off")
    else:
        print(f"[real_decoder] source_video_dir={compare_config.source_video_dir}")
        print(f"[real_decoder] compare_output_dir={compare_config.compare_output_dir}")
        print(f"[real_decoder] compare_mode={compare_config.compare_mode}")

    runtime_cache: Dict[Path, RecoveryRuntime] = {}
    model_bundle_cache: Dict[str, ModelBundle] = {}
    for latent_path in tqdm(latent_paths, desc="Real decoding latents"):
        decode_latent_file(
            latent_path=latent_path,
            input_dir=input_dir,
            output_dir=output_dir,
            root_dir=root_dir,
            metrics_dir=metrics_dir,
            recover_latent_output_dir=recover_latent_output_dir,
            cli_checkpoint_dir=args.checkpoint_dir,
            cli_base_model_path=args.base_model_path,
            cli_weight_dtype=args.weight_dtype,
            cli_num_inference_steps=args.num_inference_steps,
            seed=args.seed,
            runtime_cache=runtime_cache,
            model_bundle_cache=model_bundle_cache,
            device=device,
            compare_config=compare_config,
            transport_input_mode=args.transport_input_mode,
            vae_decode_mode=args.vae_decode_mode,
            cli_recover_overlap_latents=args.recover_overlap_latents,
            cli_recover_history_source=args.recover_history_source,
            cli_recover_init_mode=args.recover_init_mode,
            cli_recover_start_sigma=args.recover_start_sigma,
            cli_recover_low_guidance_scale=args.recover_low_guidance_scale,
        )


if __name__ == "__main__":
    main()
