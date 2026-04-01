import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm


DEFAULT_BASE_MODEL_PATH = "/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base"
DEFAULT_INPUT_DIR = "example/toy_data/videos"
DEFAULT_OUTPUT_DIR = "reconstruct/latents"
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640
DEFAULT_MAX_CHUNK_FRAMES = 81
LATENT_FORMAT_VERSION = "helios_vae_latent_v2"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".gif"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode videos into normalized Helios VAE latents.")
    parser.add_argument("--input_dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_chunk_frames", type=int, default=DEFAULT_MAX_CHUNK_FRAMES)
    return parser.parse_args()


def discover_video_paths(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    video_paths = [path for path in sorted(input_dir.rglob("*")) if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES]
    if not video_paths:
        raise FileNotFoundError(f"No video files were found in: {input_dir}")
    return video_paths


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    return device


def make_legal_chunk_limit(max_chunk_frames: int) -> int:
    if max_chunk_frames < 1:
        raise ValueError(f"max_chunk_frames must be >= 1, got {max_chunk_frames}")
    legal_limit = max_chunk_frames - ((max_chunk_frames - 1) % 4)
    if legal_limit < 1:
        raise ValueError(f"Unable to derive a legal 4n+1 chunk size from {max_chunk_frames}")
    return legal_limit


def split_frame_counts(total_frames: int, max_chunk_frames: int) -> List[int]:
    if total_frames <= 0:
        raise ValueError(f"Video must contain at least one frame, got {total_frames}")

    legal_limit = make_legal_chunk_limit(max_chunk_frames)
    start_num_chunks = total_frames % 4
    if start_num_chunks == 0:
        start_num_chunks = 4

    num_chunks = start_num_chunks
    while total_frames > num_chunks * legal_limit:
        num_chunks += 4

    shared_latent_steps = (total_frames - num_chunks) // 4
    base_steps, remainder = divmod(shared_latent_steps, num_chunks)

    chunk_lengths = []
    for idx in range(num_chunks):
        chunk_steps = base_steps + (1 if idx < remainder else 0)
        chunk_lengths.append(4 * chunk_steps + 1)

    if sum(chunk_lengths) != total_frames:
        raise RuntimeError(f"Internal chunk split error for {total_frames} frames: {chunk_lengths}")
    if any(length > legal_limit for length in chunk_lengths):
        raise RuntimeError(f"Chunk size exceeds legal limit {legal_limit}: {chunk_lengths}")
    return chunk_lengths


def build_frame_ranges(total_frames: int, max_chunk_frames: int) -> List[Tuple[int, int]]:
    chunk_lengths = split_frame_counts(total_frames, max_chunk_frames)
    frame_ranges: List[Tuple[int, int]] = []
    start = 0
    for length in chunk_lengths:
        end = start + length
        frame_ranges.append((start, end))
        start = end
    return frame_ranges


def get_video_metadata(video_path: Path) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if total_frames <= 0:
        raise RuntimeError(f"Failed to read frame count from video: {video_path}")
    if fps <= 0:
        fps = 24.0
    return total_frames, fps, height, width


def iter_video_chunks(video_path: Path, frame_ranges: Sequence[Tuple[int, int]]) -> Iterable[Tuple[np.ndarray, Tuple[int, int], int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    range_index = 0
    current_start, current_end = frame_ranges[range_index]
    current_frames: List[np.ndarray] = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frames.append(frame)
        frame_index += 1

        if frame_index == current_end:
            yield np.stack(current_frames, axis=0), (current_start, current_end), frame_index
            range_index += 1
            current_frames = []
            if range_index >= len(frame_ranges):
                break
            current_start, current_end = frame_ranges[range_index]

    cap.release()

    if range_index != len(frame_ranges):
        raise RuntimeError(
            f"Video ended early while chunking {video_path}. "
            f"Expected {len(frame_ranges)} chunks, completed {range_index}."
        )
    if current_frames:
        raise RuntimeError(f"Chunk buffering error while reading {video_path}")


def load_vae(base_model_path: str, device: torch.device) -> Tuple[AutoencoderKLWan, VideoProcessor, torch.Tensor, torch.Tensor]:
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


def encode_video(
    video_path: Path,
    input_dir: Path,
    output_dir: Path,
    video_processor: VideoProcessor,
    vae: AutoencoderKLWan,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    height: int,
    width: int,
    max_chunk_frames: int,
    device: torch.device,
) -> Dict[str, float]:
    total_frames, fps, source_height, source_width = get_video_metadata(video_path)
    frame_ranges = build_frame_ranges(total_frames, max_chunk_frames)
    relative_path = video_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    latent_chunks: List[torch.Tensor] = []
    decoded_frame_count = 0

    with torch.inference_mode():
        for chunk_frames, _, decoded_frame_count in iter_video_chunks(video_path, frame_ranges):
            chunk_images = [Image.fromarray(frame) for frame in chunk_frames]
            processed_video = video_processor.preprocess_video(chunk_images, height=height, width=width)
            processed_video = processed_video.to(device=device, dtype=vae.dtype)
            latent = vae.encode(processed_video).latent_dist.mode()
            latent = (latent - latents_mean) * latents_std
            latent_chunks.append(latent.squeeze(0).cpu().contiguous())

    if decoded_frame_count != total_frames:
        raise RuntimeError(
            f"Frame count mismatch while encoding {video_path}. "
            f"Metadata reported {total_frames}, decoded {decoded_frame_count}."
        )
    if not latent_chunks:
        raise RuntimeError(f"No latent chunks were produced for {video_path}")

    chunk_shapes = [tuple(chunk.shape) for chunk in latent_chunks]
    payload = {
        "format_version": LATENT_FORMAT_VERSION,
        "source_fps": fps,
        "latent_chunks": latent_chunks,
    }
    torch.save(payload, output_path)
    file_bytes = output_path.stat().st_size
    bpp = compute_bpp(file_bytes=file_bytes, num_frames=total_frames, width=source_width, height=source_height)
    print(
        f"[encoder] saved={output_path} "
        f"chunks={len(latent_chunks)} "
        f"latent_shapes={chunk_shapes} "
        f"file_bytes={file_bytes} "
        f"bpp={bpp:.6f}"
    )
    return {
        "file_bytes": float(file_bytes),
        "num_frames": float(total_frames),
        "source_height": float(source_height),
        "source_width": float(source_width),
        "bpp": bpp,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    video_paths = discover_video_paths(input_dir)

    print(f"[encoder] input_dir={input_dir}")
    print(f"[encoder] output_dir={output_dir}")
    print(f"[encoder] base_model_path={args.base_model_path}")
    print(f"[encoder] target_resolution={args.width}x{args.height}")

    vae, video_processor, latents_mean, latents_std = load_vae(args.base_model_path, device)
    total_file_bytes = 0
    total_source_pixels = 0

    for video_path in tqdm(video_paths, desc="Encoding videos"):
        stats = encode_video(
            video_path=video_path,
            input_dir=input_dir,
            output_dir=output_dir,
            video_processor=video_processor,
            vae=vae,
            latents_mean=latents_mean,
            latents_std=latents_std,
            height=args.height,
            width=args.width,
            max_chunk_frames=args.max_chunk_frames,
            device=device,
        )
        total_file_bytes += int(stats["file_bytes"])
        total_source_pixels += int(stats["num_frames"] * stats["source_height"] * stats["source_width"])
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dataset_bpp = compute_bpp_from_total_pixels(file_bytes=total_file_bytes, total_pixels=total_source_pixels)
    print(
        f"[encoder] summary videos={len(video_paths)} "
        f"total_file_bytes={total_file_bytes} "
        f"dataset_bpp={dataset_bpp:.6f}"
    )


if __name__ == "__main__":
    main()
