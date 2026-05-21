"""
Batch encode videos from JSON metadata files into VAE latents (multi-GPU).

Reads video paths from trainable_jsons_v1 JSON files, encodes each video
through the Wan VAE, and saves normalized latent chunks as .pt files.

Output format: helios_vae_latent_v2
  - format_version: str
  - source_fps: float
  - latent_chunks: List[Tensor]  # each [C, T_chunk, H, W]
"""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from PIL import Image
from tqdm import tqdm

DEFAULT_BASE_MODEL_PATH = "/app1/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base"
DEFAULT_JSON_DIR = "/data_old/gemini/gemini-sharedata/space/qdrnbvyr7uio/yifq/yanjq/video_data/trainable_jsons_v1"
DEFAULT_OUTPUT_DIR = "/data/heli/code/tele_Gen/dataset_10w"
DEFAULT_HEIGHT = 384
DEFAULT_WIDTH = 640
DEFAULT_MAX_CHUNK_FRAMES = None
DEFAULT_MAX_ENCODE_FRAMES = None
LATENT_FORMAT_VERSION = "helios_vae_latent_v2"
PATH_PREFIX_FROM = "/workspace/gemini/gemini-sharedata"
PATH_PREFIX_TO = "/data_old/gemini/gemini-sharedata"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch encode videos from JSON metadata into VAE latents.")
    parser.add_argument("--json_dir", type=Path, default=Path(DEFAULT_JSON_DIR))
    parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument(
        "--max_chunk_frames",
        type=int,
        default=DEFAULT_MAX_CHUNK_FRAMES,
        help=(
            "Optional 4n+1 chunk size limit. Defaults to None, which encodes each video "
            "as one variable-length no-chunk latent."
        ),
    )
    parser.add_argument(
        "--max_encode_frames",
        type=int,
        default=DEFAULT_MAX_ENCODE_FRAMES,
        help=(
            "Optional maximum frames to encode from each source video before snapping down "
            "to a legal 4n+1 length. Defaults to None, so the whole source clip is used."
        ),
    )
    parser.add_argument("--max_videos", type=int, default=10000)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def fix_video_path(video_path: str) -> str:
    if video_path.startswith(PATH_PREFIX_FROM):
        return PATH_PREFIX_TO + video_path[len(PATH_PREFIX_FROM):]
    return video_path


def make_output_name(video_path: str) -> str:
    """Generate unique output filename from video path.

    Handles patterns like:
      .../58594.mp4/0.mp4         -> 58594_0.pt
      .../moviename.mp4/81.mp4    -> moviename_81.pt
    """
    p = Path(video_path)
    parent_name = p.parent.name
    if parent_name.endswith(".mp4"):
        parent_name = parent_name[:-4]
    return f"{parent_name}_{p.stem}.pt"


def load_clip_metadata(json_dir: Path, max_videos: int) -> List[Dict]:
    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {json_dir}")

    all_clips = []
    for jf in json_files:
        data = json.load(open(jf))
        clips = data.get("clips", [])
        all_clips.extend(clips)
        if len(all_clips) >= max_videos:
            break

    return all_clips[:max_videos]


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
    frame_ranges = []
    start = 0
    for length in chunk_lengths:
        end = start + length
        frame_ranges.append((start, end))
        start = end
    return frame_ranges


def resolve_encode_frame_count(total_frames: int, max_encode_frames: Optional[int]) -> int:
    if total_frames <= 0:
        raise ValueError(f"Video must contain at least one frame, got {total_frames}")
    if max_encode_frames is not None and max_encode_frames < 1:
        raise ValueError(f"max_encode_frames must be >= 1 when set, got {max_encode_frames}")

    frame_count = total_frames if max_encode_frames is None else min(total_frames, max_encode_frames)
    legal_frame_count = frame_count - ((frame_count - 1) % 4)
    if legal_frame_count < 1:
        raise ValueError(
            f"Unable to derive a legal 4n+1 encode length from total_frames={total_frames}, "
            f"max_encode_frames={max_encode_frames}"
        )
    return legal_frame_count


def resolve_frame_ranges(
    total_frames: int,
    max_encode_frames: Optional[int],
    max_chunk_frames: Optional[int],
) -> List[Tuple[int, int]]:
    encoded_frames = resolve_encode_frame_count(total_frames, max_encode_frames)
    if max_chunk_frames is None:
        return [(0, encoded_frames)]
    return build_frame_ranges(encoded_frames, max_chunk_frames)


def get_video_metadata(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(video_path)
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


def iter_video_chunks(video_path: str, frame_ranges: List[Tuple[int, int]]):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    range_index = 0
    current_start, current_end = frame_ranges[range_index]
    current_frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_frames.append(frame)
        frame_index += 1

        if frame_index == current_end:
            yield np.stack(current_frames, axis=0), (current_start, current_end)
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


def load_vae(base_model_path: str, device: torch.device):
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


def encode_single_video(
    video_path: str,
    output_path: Path,
    video_processor: VideoProcessor,
    vae: AutoencoderKLWan,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    height: int,
    width: int,
    max_chunk_frames: Optional[int],
    max_encode_frames: Optional[int],
    device: torch.device,
) -> Optional[Dict]:
    try:
        total_frames, fps, source_height, source_width = get_video_metadata(video_path)
    except Exception as e:
        return {"error": f"metadata: {e}"}

    try:
        frame_ranges = resolve_frame_ranges(total_frames, max_encode_frames, max_chunk_frames)
        encoded_frames = frame_ranges[-1][1]
    except Exception as e:
        return {"error": f"frame_ranges: {e}"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    latent_chunks = []

    try:
        with torch.inference_mode():
            for chunk_frames, _ in iter_video_chunks(video_path, frame_ranges):
                chunk_images = [Image.fromarray(frame) for frame in chunk_frames]
                processed_video = video_processor.preprocess_video(chunk_images, height=height, width=width)
                processed_video = processed_video.to(device=device, dtype=vae.dtype)
                latent = vae.encode(processed_video).latent_dist.mode()
                latent = (latent - latents_mean) * latents_std
                latent_chunks.append(latent.squeeze(0).cpu().contiguous())
    except Exception as e:
        return {"error": f"encode: {e}"}

    if not latent_chunks:
        return {"error": "no latent chunks produced"}

    payload = {
        "format_version": LATENT_FORMAT_VERSION,
        "source_fps": fps,
        "source_num_frames": encoded_frames,
        "source_total_frames": total_frames,
        "encoded_num_frames": encoded_frames,
        "source_width": source_width,
        "source_height": source_height,
        "encoded_width": width,
        "encoded_height": height,
        "latent_chunks": latent_chunks,
    }
    torch.save(payload, output_path)
    file_bytes = output_path.stat().st_size
    total_pixels = encoded_frames * source_width * source_height
    bpp = file_bytes * 8.0 / total_pixels if total_pixels > 0 else 0.0

    return {
        "file_bytes": file_bytes,
        "num_frames": encoded_frames,
        "source_total_frames": total_frames,
        "source_height": source_height,
        "source_width": source_width,
        "bpp": bpp,
    }


def gpu_worker(
    gpu_id: int,
    clips: List[Dict],
    output_dir: Path,
    base_model_path: str,
    height: int,
    width: int,
    max_chunk_frames: Optional[int],
    max_encode_frames: Optional[int],
    result_queue: multiprocessing.Queue,
):
    """Worker process: encode a shard of clips on one GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    vae, video_processor, latents_mean, latents_std = load_vae(base_model_path, device)

    success_count = 0
    error_count = 0
    total_file_bytes = 0
    total_source_pixels = 0
    errors = []

    desc = f"GPU {gpu_id}"
    for clip in tqdm(clips, desc=desc, position=gpu_id, leave=True):
        video_path = fix_video_path(clip["video_path"])
        output_name = make_output_name(video_path)
        output_path = output_dir / output_name

        stats = encode_single_video(
            video_path=video_path,
            output_path=output_path,
            video_processor=video_processor,
            vae=vae,
            latents_mean=latents_mean,
            latents_std=latents_std,
            height=height,
            width=width,
            max_chunk_frames=max_chunk_frames,
            max_encode_frames=max_encode_frames,
            device=device,
        )

        if stats is None or "error" in stats:
            error_count += 1
            err_msg = stats.get("error", "unknown") if stats else "unknown"
            errors.append(f"{video_path}: {err_msg}")
        else:
            success_count += 1
            total_file_bytes += stats["file_bytes"]
            total_source_pixels += stats["num_frames"] * stats["source_height"] * stats["source_width"]

        torch.cuda.empty_cache()

    result_queue.put({
        "gpu_id": gpu_id,
        "success": success_count,
        "errors": error_count,
        "total_file_bytes": total_file_bytes,
        "total_source_pixels": total_source_pixels,
        "error_log": errors,
    })


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    print(f"[batch_encoder] json_dir={args.json_dir}")
    print(f"[batch_encoder] output_dir={output_dir}")
    print(f"[batch_encoder] base_model_path={args.base_model_path}")
    print(f"[batch_encoder] target_resolution={args.width}x{args.height}")
    print(f"[batch_encoder] max_chunk_frames={args.max_chunk_frames}")
    print(f"[batch_encoder] max_encode_frames={args.max_encode_frames}")
    print(f"[batch_encoder] max_videos={args.max_videos}")
    print(f"[batch_encoder] num_gpus={num_gpus}")

    # Load clip metadata
    clips = load_clip_metadata(args.json_dir, args.max_videos)
    print(f"[batch_encoder] loaded {len(clips)} clip metadata entries")

    # Filter already encoded
    if args.skip_existing:
        remaining = []
        for clip in clips:
            video_path = fix_video_path(clip["video_path"])
            output_name = make_output_name(video_path)
            output_path = output_dir / output_name
            if not output_path.exists():
                remaining.append(clip)
        print(f"[batch_encoder] {len(clips) - len(remaining)} already encoded, {len(remaining)} remaining")
        clips = remaining

    if not clips:
        print("[batch_encoder] nothing to encode, exiting")
        return

    # Split clips across GPUs
    shards = [[] for _ in range(num_gpus)]
    for i, clip in enumerate(clips):
        shards[i % num_gpus].append(clip)
    for i, shard in enumerate(shards):
        print(f"  GPU {i}: {len(shard)} videos")

    # Launch worker processes
    result_queue = multiprocessing.Queue()
    processes = []
    for gpu_id in range(num_gpus):
        if not shards[gpu_id]:
            continue
        p = multiprocessing.Process(
            target=gpu_worker,
            args=(
                gpu_id,
                shards[gpu_id],
                output_dir,
                args.base_model_path,
                args.height,
                args.width,
                args.max_chunk_frames,
                args.max_encode_frames,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers
    for p in processes:
        p.join()

    # Collect results
    total_success = 0
    total_errors = 0
    total_file_bytes = 0
    total_source_pixels = 0
    all_errors = []

    while not result_queue.empty():
        result = result_queue.get()
        total_success += result["success"]
        total_errors += result["errors"]
        total_file_bytes += result["total_file_bytes"]
        total_source_pixels += result["total_source_pixels"]
        all_errors.extend(result["error_log"])

    print(f"\n[batch_encoder] === Summary ===")
    print(f"[batch_encoder] total_clips: {len(clips)}")
    print(f"[batch_encoder] success: {total_success}")
    print(f"[batch_encoder] errors: {total_errors}")

    if total_source_pixels > 0:
        dataset_bpp = total_file_bytes * 8.0 / total_source_pixels
        print(f"[batch_encoder] total_file_bytes: {total_file_bytes}")
        print(f"[batch_encoder] dataset_bpp: {dataset_bpp:.6f}")

    if all_errors:
        error_log_path = output_dir / "encoding_errors.log"
        with open(error_log_path, "w") as f:
            f.write("\n".join(all_errors))
        print(f"[batch_encoder] error log saved to {error_log_path}")


if __name__ == "__main__":
    main()
