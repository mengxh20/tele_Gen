import io
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor


DEFAULT_BASE_MODEL_PATH = os.environ.get(
    "HELIOS_BASE_MODEL_PATH",
    "/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base",
).strip()
LATENT_FORMAT_V1 = "helios_vae_latent_v1"
LATENT_FORMAT_V2 = "helios_vae_latent_v2"
LOW_LATENT_FORMAT_V1 = "helios_low_latent_v1"
LOW_LATENT_FORMAT_V2 = "helios_low_latent_v2"
LOW_LATENT_FORMAT_V3 = "helios_low_latent_v3"
ModelBundle = Tuple[AutoencoderKLWan, VideoProcessor, torch.Tensor, torch.Tensor]
_RESOLUTION_PATTERN = re.compile(r"^(?P<width>\d+)x(?P<height>\d+)$")


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    return device


def discover_latent_paths(input_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    latent_paths = sorted(path for path in input_dir.rglob("*.pt") if path.is_file())
    if max_files is not None:
        latent_paths = latent_paths[:max_files]
    if not latent_paths:
        raise FileNotFoundError(f"No latent files were found in: {input_dir}")
    return latent_paths


def validate_payload(payload: Dict, latent_path: Path) -> None:
    format_version = payload.get("format_version")
    if format_version == LATENT_FORMAT_V1:
        required_keys = {
            "format_version",
            "source_path",
            "source_fps",
            "source_num_frames",
            "source_height",
            "source_width",
            "processed_height",
            "processed_width",
            "chunk_frame_ranges",
            "latent_chunks",
            "latent_dtype",
            "latent_normalized",
            "model_path",
        }
    elif format_version == LATENT_FORMAT_V2:
        required_keys = {
            "format_version",
            "source_fps",
            "latent_chunks",
        }
    else:
        raise ValueError(
            f"Unsupported latent format in {latent_path}: {format_version}. "
            f"Only {LATENT_FORMAT_V1} and {LATENT_FORMAT_V2} are supported."
        )

    missing_keys = sorted(required_keys - payload.keys())
    if missing_keys:
        raise KeyError(f"Latent file {latent_path} is missing keys: {missing_keys}")
    if format_version == LATENT_FORMAT_V1 and not payload["latent_normalized"]:
        raise ValueError(f"Latent file {latent_path} is not marked as normalized.")

    latent_chunks = payload["latent_chunks"]
    if not isinstance(latent_chunks, list) or not latent_chunks:
        raise ValueError(f"Latent file {latent_path} does not contain any latent chunks.")

    chunk_frame_ranges = payload.get("chunk_frame_ranges")
    if chunk_frame_ranges is not None and len(latent_chunks) != len(chunk_frame_ranges):
        raise ValueError(
            f"Latent file {latent_path} has mismatched chunk metadata: "
            f"{len(latent_chunks)} latent chunks vs {len(chunk_frame_ranges)} frame ranges."
        )


def load_payload(latent_path: Path) -> Dict:
    try:
        return torch.load(latent_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(latent_path, map_location="cpu")


def load_vae_bundle(base_model_path: str, device: torch.device) -> ModelBundle:
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    if hasattr(vae, "enable_slicing"):
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    vae.eval()
    vae.requires_grad_(False)
    vae = vae.to(device)

    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(1, vae.config.z_dim, 1, 1, 1)
    )
    spatial_scale_factor = 2 ** (len(vae.config.get("dim_mult", [])) - 1)
    if spatial_scale_factor <= 0:
        spatial_scale_factor = 8
    video_processor = VideoProcessor(vae_scale_factor=spatial_scale_factor)
    return vae, video_processor, latents_mean, latents_std


def compute_bpp(file_bytes: int, num_frames: int, width: int, height: int) -> float:
    total_pixels = num_frames * width * height
    if total_pixels <= 0:
        raise ValueError(
            f"Unable to compute bpp with num_frames={num_frames}, width={width}, height={height}."
        )
    return file_bytes * 8.0 / total_pixels


def compute_bpp_from_total_pixels(file_bytes: int, total_pixels: int) -> float:
    if total_pixels <= 0:
        raise ValueError(f"Unable to compute bpp with total_pixels={total_pixels}.")
    return file_bytes * 8.0 / total_pixels


def serialize_payload_num_bytes(payload: Dict) -> int:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.tell()


def infer_source_num_frames(latent_chunks: Sequence[torch.Tensor], payload: Dict) -> int:
    source_num_frames = payload.get("source_num_frames")
    if source_num_frames is not None:
        return int(source_num_frames)
    return sum(infer_chunk_num_frames_from_latent_length(chunk.shape[1]) for chunk in latent_chunks)


def parse_source_resolution_from_filename(path: Path) -> Tuple[int, int]:
    for part in reversed(path.stem.split("_")):
        match = _RESOLUTION_PATTERN.fullmatch(part)
        if match is not None:
            return int(match.group("width")), int(match.group("height"))
    raise ValueError(
        f"Unable to infer source resolution from {path.name}. Expected a filename token like 1280x720."
    )


def resolve_source_resolution(payload: Dict, latent_path: Path) -> Tuple[int, int]:
    source_width = payload.get("source_width")
    source_height = payload.get("source_height")
    if source_width is not None and source_height is not None:
        width = int(source_width)
        height = int(source_height)
        if width > 0 and height > 0:
            return width, height
    return parse_source_resolution_from_filename(latent_path)


def infer_chunk_num_frames_from_latent_length(latent_length: int) -> int:
    if latent_length < 1:
        raise ValueError(f"Latent chunk length must be >= 1, got {latent_length}.")
    return (latent_length - 1) * 4 + 1


def infer_chunk_frame_ranges(
    latent_chunks: Sequence[torch.Tensor],
    payload: Dict,
) -> List[Tuple[int, int]]:
    chunk_frame_ranges = payload.get("chunk_frame_ranges")
    if chunk_frame_ranges is not None:
        return [tuple(int(value) for value in frame_range) for frame_range in chunk_frame_ranges]

    chunk_frame_counts = [infer_chunk_num_frames_from_latent_length(chunk.shape[1]) for chunk in latent_chunks]
    return build_chunk_frame_ranges(chunk_frame_counts)


def build_chunk_frame_ranges(chunk_frame_counts: Sequence[int]) -> List[Tuple[int, int]]:
    frame_ranges: List[Tuple[int, int]] = []
    start = 0
    for count in chunk_frame_counts:
        end = start + int(count)
        frame_ranges.append((start, end))
        start = end
    return frame_ranges


def flatten_latent_chunks(latent_chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    if not latent_chunks:
        raise ValueError("Expected at least one latent chunk.")
    return torch.cat([chunk.float().contiguous() for chunk in latent_chunks], dim=1)


def split_full_latents(full_latents: torch.Tensor, chunk_lengths: Sequence[int]) -> List[torch.Tensor]:
    restored_chunks: List[torch.Tensor] = []
    start = 0
    for length in chunk_lengths:
        end = start + int(length)
        restored_chunks.append(full_latents[:, start:end].contiguous())
        start = end

    if start != full_latents.shape[1]:
        raise ValueError(
            f"Chunk lengths do not match full latent length: used {start}, total {full_latents.shape[1]}."
        )
    return restored_chunks


def resolve_model_path(
    payload: Dict,
    cli_model_path: Optional[str],
    default_model_path: Optional[str] = None,
) -> str:
    model_path = cli_model_path or payload.get("model_path") or default_model_path
    if not model_path:
        raise ValueError(
            "Latent payload does not include a usable model_path. "
            "Please pass --base_model_path explicitly."
        )
    return str(model_path)
