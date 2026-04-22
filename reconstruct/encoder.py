import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.distributed as dist
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


@dataclass(frozen=True)
class DistributedContext:
    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode videos into normalized Helios VAE latents.")
    parser.add_argument("--input_dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--input_list", type=Path, default=None)
    parser.add_argument("--start_index", type=int, default=1)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--path_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_chunk_frames", type=int, default=DEFAULT_MAX_CHUNK_FRAMES)
    parser.add_argument("--skip_missing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def is_supported_video_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def discover_video_paths(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    video_paths = [path for path in sorted(input_dir.rglob("*")) if is_supported_video_path(path)]
    if not video_paths:
        raise FileNotFoundError(f"No video files were found in: {input_dir}")
    return video_paths


def load_video_paths_from_list(input_list: Path) -> List[Tuple[int, int, Path]]:
    resolved_list = input_list.resolve()
    if not resolved_list.exists():
        raise FileNotFoundError(f"Input list does not exist: {resolved_list}")

    entries: List[Tuple[int, int, Path]] = []
    with resolved_list.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            candidate = raw_line.strip()
            if not candidate or candidate.startswith("#"):
                continue
            path = Path(candidate).expanduser()
            if not path.is_absolute():
                path = (resolved_list.parent / path).resolve()
            else:
                path = path.resolve()
            entry_index = len(entries) + 1
            entries.append((entry_index, line_number, path))

    if not entries:
        raise FileNotFoundError(f"No usable video paths were found in: {resolved_list}")
    return entries


def validate_list_slice(start_index: int, end_index: Optional[int], total_entries: int) -> Tuple[int, int]:
    if start_index < 1:
        raise ValueError(f"start_index must be >= 1, got {start_index}")

    resolved_end = total_entries if end_index is None else end_index
    if resolved_end < start_index:
        raise ValueError(f"end_index must be >= start_index, got start_index={start_index}, end_index={resolved_end}")
    if start_index > total_entries:
        raise ValueError(f"start_index={start_index} exceeds available entries={total_entries}")
    if resolved_end > total_entries:
        raise ValueError(f"end_index={resolved_end} exceeds available entries={total_entries}")
    return start_index, resolved_end


def infer_input_root(video_paths: Sequence[Path], path_root: Optional[Path]) -> Path:
    if not video_paths:
        raise ValueError("Expected at least one video path when inferring input root.")

    if path_root is not None:
        resolved_root = path_root.resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"path_root does not exist: {resolved_root}")
        for video_path in video_paths:
            try:
                video_path.relative_to(resolved_root)
            except ValueError as exc:
                raise ValueError(f"Video path {video_path} is not under path_root={resolved_root}") from exc
        return resolved_root

    if len(video_paths) == 1:
        return video_paths[0].parent

    common_path = Path(os.path.commonpath([str(path) for path in video_paths]))
    if common_path.is_file():
        return common_path.parent
    return common_path


def discover_video_paths_from_list(
    input_list: Path,
    start_index: int,
    end_index: Optional[int],
    path_root: Optional[Path],
    skip_missing: bool,
) -> Tuple[Path, List[Path], int, int]:
    entries = load_video_paths_from_list(input_list)
    start_index, end_index = validate_list_slice(start_index=start_index, end_index=end_index, total_entries=len(entries))
    selected_entries = entries[start_index - 1:end_index]

    selected_paths: List[Path] = []
    missing_paths: List[Tuple[int, int, Path]] = []
    invalid_paths: List[Tuple[int, int, Path]] = []
    for entry_index, line_number, path in selected_entries:
        if not path.exists():
            missing_paths.append((entry_index, line_number, path))
            continue
        if not is_supported_video_path(path):
            invalid_paths.append((entry_index, line_number, path))
            continue
        selected_paths.append(path)

    if invalid_paths:
        preview = ", ".join(
            f"entry={entry_index} line={line_number} path={path}"
            for entry_index, line_number, path in invalid_paths[:5]
        )
        raise ValueError(f"Found unsupported paths in input list: {preview}")

    if missing_paths and not skip_missing:
        preview = ", ".join(
            f"entry={entry_index} line={line_number} path={path}"
            for entry_index, line_number, path in missing_paths[:5]
        )
        raise FileNotFoundError(
            f"Found {len(missing_paths)} missing paths in input list. "
            f"Use --skip_missing to ignore them. Examples: {preview}"
        )

    if not selected_paths:
        raise FileNotFoundError("No valid video paths remain after applying the requested list slice.")

    input_root = infer_input_root(selected_paths, path_root=path_root)
    return input_root, selected_paths, len(selected_entries), len(missing_paths)


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    return device


def init_distributed_context(device_arg: str) -> Tuple[DistributedContext, torch.device]:
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
            raise RuntimeError("CUDA distributed encoding was requested but CUDA is not available.")
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
    if distributed_context.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed_context: DistributedContext) -> None:
    if distributed_context.is_distributed and dist.is_initialized():
        if dist.get_backend() == "nccl":
            dist.barrier(device_ids=[distributed_context.local_rank])
        else:
            dist.barrier()


def shard_video_paths(video_paths: Sequence[Path], distributed_context: DistributedContext) -> List[Path]:
    return list(video_paths[distributed_context.rank :: distributed_context.world_size])


def reduce_encoding_totals(
    total_file_bytes: int,
    total_source_pixels: int,
    encoded_videos: int,
    device: torch.device,
    distributed_context: DistributedContext,
) -> Tuple[int, int, int]:
    tensor_device = device if device.type == "cuda" else torch.device("cpu")
    totals = torch.tensor(
        [float(total_file_bytes), float(total_source_pixels), float(encoded_videos)],
        dtype=torch.float64,
        device=tensor_device,
    )
    if distributed_context.is_distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return int(totals[0].item()), int(totals[1].item()), int(totals[2].item())


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
    input_root: Path,
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
    relative_path = video_path.relative_to(input_root)
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
        "source_path": str(video_path),
        "source_fps": fps,
        "source_num_frames": total_frames,
        "source_height": source_height,
        "source_width": source_width,
        "processed_height": height,
        "processed_width": width,
        "chunk_frame_ranges": frame_ranges,
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
    output_dir = args.output_dir.resolve()
    distributed_context, device = init_distributed_context(args.device)

    try:
        if args.input_list is not None:
            input_root, video_paths, selected_entries, missing_entries = discover_video_paths_from_list(
                input_list=args.input_list,
                start_index=args.start_index,
                end_index=args.end_index,
                path_root=args.path_root,
                skip_missing=args.skip_missing,
            )
        else:
            input_root = args.input_dir.resolve()
            video_paths = discover_video_paths(input_root)
            selected_entries = len(video_paths)
            missing_entries = 0

        assigned_video_paths = shard_video_paths(video_paths, distributed_context)

        if distributed_context.is_main_process:
            print("start!")
            if args.input_list is not None:
                print(f"[encoder] input_list={args.input_list.resolve()}")
                print(f"[encoder] selected_entries={selected_entries}")
                print(f"[encoder] valid_videos={len(video_paths)}")
                if missing_entries:
                    print(f"[encoder] skipped_missing={missing_entries}")
            else:
                print(f"[encoder] input_dir={input_root}")
            print(f"[encoder] input_root={input_root}")
            print(f"[encoder] output_dir={output_dir}")
            print(f"[encoder] base_model_path={args.base_model_path}")
            print(f"[encoder] target_resolution={args.width}x{args.height}")
            print(f"[encoder] num_videos={len(video_paths)}")
            print(f"[encoder] first_video={video_paths[0]}")
            print(f"[encoder] last_video={video_paths[-1]}")
            print(
                f"[encoder] device={device} world_size={distributed_context.world_size} "
                f"distributed={distributed_context.is_distributed}"
            )
            output_dir.mkdir(parents=True, exist_ok=True)

        distributed_barrier(distributed_context)

        print(
            f"[encoder][rank={distributed_context.rank}] assigned_videos={len(assigned_video_paths)} "
            f"device={device}"
        )
        if assigned_video_paths:
            print(f"[encoder][rank={distributed_context.rank}] first_assigned={assigned_video_paths[0]}")
            print(f"[encoder][rank={distributed_context.rank}] last_assigned={assigned_video_paths[-1]}")

        if args.dry_run:
            print(f"[encoder][rank={distributed_context.rank}] dry_run complete")
            distributed_barrier(distributed_context)
            return

        vae, video_processor, latents_mean, latents_std = load_vae(args.base_model_path, device)
        total_file_bytes = 0
        total_source_pixels = 0

        progress_bar = tqdm(
            assigned_video_paths,
            desc=f"Encoding videos rank={distributed_context.rank}",
            disable=distributed_context.is_distributed and not distributed_context.is_main_process,
        )
        for video_path in progress_bar:
            stats = encode_video(
                video_path=video_path,
                input_root=input_root,
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

        local_videos = len(assigned_video_paths)
        print(
            f"[encoder][rank={distributed_context.rank}] local_summary "
            f"videos={local_videos} total_file_bytes={total_file_bytes}"
        )

        reduced_file_bytes, reduced_source_pixels, reduced_videos = reduce_encoding_totals(
            total_file_bytes=total_file_bytes,
            total_source_pixels=total_source_pixels,
            encoded_videos=local_videos,
            device=device,
            distributed_context=distributed_context,
        )

        if distributed_context.is_main_process:
            dataset_bpp = compute_bpp_from_total_pixels(
                file_bytes=reduced_file_bytes,
                total_pixels=reduced_source_pixels,
            )
            print(
                f"[encoder] summary videos={reduced_videos} "
                f"total_file_bytes={reduced_file_bytes} "
                f"dataset_bpp={dataset_bpp:.6f}"
            )

        distributed_barrier(distributed_context)
    finally:
        cleanup_distributed_context(distributed_context)


if __name__ == "__main__":
    main()
