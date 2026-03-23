#!/usr/bin/env python3
"""
Visualize Helios stage1 latent dataset files.

# 传目录和传单个.pt文件都行
Examples:
  python new/visualize_helios_stage1_dataset.py
  python new/visualize_helios_stage1_dataset.py /path/to/sample.pt --decode-sections 0
  python new/visualize_helios_stage1_dataset.py /gemini/platform/public/luojx/team/mengxh/codes/Helios/example/toy_data/latents_short/toy_data \
      --base-model-path /gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base \
      --decode-sections 1 --save-video
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


DEFAULT_DATASET_ROOT = (
    "/gemini/platform/public/luojx/team/mengxh/MODELS/HeliosBench-Weights/demo_data/ultravideo-long"
)
DEFAULT_BASE_MODEL_PATH = "/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base"


def parse_decode_sections(value: str) -> int | None:
    if value.lower() == "all":
        return None

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("`--decode-sections` must be a non-negative integer or `all`.") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError("`--decode-sections` must be a non-negative integer or `all`.")

    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Helios stage1 `.pt` dataset files. "
            "This script can inspect file contents, save preview figures, and optionally decode `vae_latent`."
        )
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        default=DEFAULT_DATASET_ROOT,
        help="Path to a single `.pt` file or a directory containing `.pt` files.",
    )
    parser.add_argument(
        "--output-dir",
        default="new/visualizations",
        help="Directory used to save summaries and preview assets.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=4,
        help="How many `.pt` files to process when `input_path` is a directory.",
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help=(
            "Helios base model path used to load the VAE for latent decoding. "
            "If omitted, the script only exports metadata and latent-space previews. "
            f"The training config in this repo uses: {DEFAULT_BASE_MODEL_PATH}"
        ),
    )
    parser.add_argument(
        "--decode-sections",
        type=parse_decode_sections,
        default="all",
        help="How many leading latent sections to decode. Use `0` to skip decoding, or `all` to decode every section.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="FPS used when writing decoded preview videos.",
    )
    parser.add_argument(  # 预览多少张图，对生成的视频们没有影响
        "--num-preview-frames",
        type=int,
        default=12,
        help="How many frames to show in contact sheets.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for VAE decoding.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Try to save decoded MP4/GIF previews when latent decoding is enabled.",
    )
    parser.add_argument(
        "--latent-rgb-preview",
        action="store_true",
        help="Also export a pseudo-RGB PCA projection of the latent frames.",
    )
    parser.add_argument(
        "--print-prompt-chars",
        type=int,
        default=200,
        help="Maximum number of prompt characters echoed to stdout.",
    )
    parser.add_argument(
        "--source-video-dir",
        default=None,
        help="Directory containing source videos used for reconstruction metrics such as LPIPS.",
    )
    parser.add_argument(
        "--filter-json",
        default=None,
        help="Optional filter json that stores source `path`, `cut`, and `crop` metadata for each sample.",
    )
    parser.add_argument(
        "--compute-metrics",
        action="store_true",
        help="Compute compression and reconstruction metrics. This adds latent bpp and LPIPS outputs.",
    )
    parser.add_argument(
        "--metric-batch-size",
        type=int,
        default=8,
        help="Batch size used when computing frame-wise LPIPS.",
    )
    parser.add_argument(
        "--lpips-net",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by the LPIPS metric.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")
    return device_arg


def resolve_pt_files(input_path: Path, max_files: int) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix != ".pt":
            raise ValueError(f"Expected a `.pt` file, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Path not found: {input_path}")

    pt_files = sorted(input_path.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No `.pt` files found under: {input_path}")
    return pt_files[:max_files]


def parse_sample_name(file_path: Path) -> dict[str, Any]:
    parts = file_path.stem.split("_")
    if len(parts) < 4:
        return {
            "uttid": file_path.stem,
            "num_frame": None,
            "height": None,
            "width": None,
            "source_video_stem": None,
            "cut_start": None,
            "cut_end": None,
        }

    try:
        uttid = "_".join(parts[:-3])
        num_frame = int(parts[-3])
        height = int(parts[-2])
        width = int(parts[-1])
        source_video_stem = None
        cut_start = None
        cut_end = None
        if len(parts) >= 5 and "-" in parts[-4]:
            cut_tokens = parts[-4].split("-", maxsplit=1)
            if len(cut_tokens) == 2:
                cut_start = int(cut_tokens[0])
                cut_end = int(cut_tokens[1])
                source_video_stem = "_".join(parts[:-4])
        return {
            "uttid": uttid,
            "num_frame": num_frame,
            "height": height,
            "width": width,
            "source_video_stem": source_video_stem,
            "cut_start": cut_start,
            "cut_end": cut_end,
        }
    except ValueError:
        return {
            "uttid": file_path.stem,
            "num_frame": None,
            "height": None,
            "width": None,
            "source_video_stem": None,
            "cut_start": None,
            "cut_end": None,
        }


def to_serializable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if torch.is_tensor(value):
        tensor = value.detach().to(torch.float32).cpu()
        result = {
            "type": "tensor",
            "shape": list(tensor.shape),
            "dtype": str(value.dtype),
            "numel": int(tensor.numel()),
        }
        if tensor.numel() > 0:
            result.update(
                {
                    "min": float(tensor.min().item()),
                    "max": float(tensor.max().item()),
                    "mean": float(tensor.mean().item()),
                    "std": float(tensor.std(unbiased=False).item()),
                }
            )
        return result
    if isinstance(value, Image.Image):
        return {"type": "PIL.Image", "mode": value.mode, "size": list(value.size)}
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def make_contact_sheet(
    frames: list[np.ndarray],
    output_path: Path,
    title: str,
    max_cols: int = 4,
) -> None:
    if not frames:
        return

    cols = min(max_cols, len(frames))
    rows = math.ceil(len(frames) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if isinstance(axes, np.ndarray):
        axes_list = axes.flatten()
    else:
        axes_list = [axes]

    for axis, frame in zip(axes_list, frames):
        if frame.ndim == 2:
            axis.imshow(frame, cmap="viridis")
        else:
            axis.imshow(frame)
        axis.axis("off")

    for axis in axes_list[len(frames) :]:
        axis.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def evenly_spaced_indices(length: int, num_items: int) -> list[int]:
    if length <= 0:
        return []
    if num_items >= length:
        return list(range(length))
    return np.linspace(0, length - 1, num=num_items, dtype=int).tolist()


def flatten_latent_sections(vae_latent: torch.Tensor) -> torch.Tensor:
    if vae_latent.ndim != 5:
        raise ValueError(f"Expected `vae_latent` to have 5 dims, got: {tuple(vae_latent.shape)}")
    return vae_latent.permute(0, 2, 1, 3, 4).reshape(-1, vae_latent.shape[1], vae_latent.shape[3], vae_latent.shape[4])


def save_latent_heatmap_preview(vae_latent: torch.Tensor, output_path: Path, num_preview_frames: int) -> None:
    flat_latents = flatten_latent_sections(vae_latent).to(torch.float32)
    frame_indices = evenly_spaced_indices(flat_latents.shape[0], num_preview_frames)
    heatmaps = []
    for idx in frame_indices:
        heatmap = flat_latents[idx].abs().mean(dim=0).cpu().numpy()
        heatmaps.append(heatmap)
    make_contact_sheet(heatmaps, output_path, title="Latent Magnitude Preview")


def pca_project_latents_to_rgb(flat_latents: torch.Tensor, max_points: int = 20000) -> np.ndarray:
    frames, channels, height, width = flat_latents.shape
    pixels = flat_latents.permute(0, 2, 3, 1).reshape(-1, channels).to(torch.float32)

    if pixels.shape[0] > max_points:
        sampled_indices = torch.linspace(0, pixels.shape[0] - 1, steps=max_points).long()
        sampled_pixels = pixels[sampled_indices]
    else:
        sampled_pixels = pixels

    sampled_pixels = sampled_pixels - sampled_pixels.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(sampled_pixels, q=3)

    projected = pixels @ v[:, :3]
    projected = projected.reshape(frames, height, width, 3)

    projected_np = projected.cpu().numpy()
    low = np.percentile(projected_np, 1, axis=(0, 1, 2), keepdims=True)
    high = np.percentile(projected_np, 99, axis=(0, 1, 2), keepdims=True)
    projected_np = np.clip((projected_np - low) / np.maximum(high - low, 1e-6), 0.0, 1.0)
    return (projected_np * 255.0).astype(np.uint8)


def save_latent_rgb_preview(vae_latent: torch.Tensor, output_path: Path, num_preview_frames: int) -> None:
    flat_latents = flatten_latent_sections(vae_latent)
    frame_indices = evenly_spaced_indices(flat_latents.shape[0], num_preview_frames)
    rgb_frames = pca_project_latents_to_rgb(flat_latents[frame_indices])
    make_contact_sheet(list(rgb_frames), output_path, title="Latent PCA RGB Preview")


def write_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def save_prompt_text(prompt_raw: str | None, output_path: Path) -> None:
    if prompt_raw is None:
        return
    output_path.write_text(prompt_raw, encoding="utf-8")


def save_first_frame_image(first_frame_image: Any, output_path: Path) -> None:
    if isinstance(first_frame_image, Image.Image):
        first_frame_image.save(output_path)


def load_filter_records(filter_json_path: str | None) -> dict[str, dict[str, Any]]:
    if filter_json_path is None:
        return {}

    filter_path = Path(filter_json_path)
    if not filter_path.exists():
        raise FileNotFoundError(f"Filter json not found: {filter_path}")

    raw_records = json.loads(filter_path.read_text(encoding="utf-8"))
    record_map: dict[str, dict[str, Any]] = {}
    for record in raw_records:
        video_rel_path = record.get("path")
        cut = record.get("cut")
        if not video_rel_path or not isinstance(cut, list) or len(cut) != 2:
            continue

        source_stem = Path(video_rel_path).stem
        cut_start = int(cut[0])
        cut_end = int(cut[1])
        key = f"{source_stem}_{cut_start}-{cut_end}"
        record_map[key] = {
            "source_stem": source_stem,
            "relative_path": video_rel_path,
            "cut_start": cut_start,
            "cut_end": cut_end,
            "crop": [int(v) for v in record.get("crop", [])],
            "fps": float(record.get("fps", 0.0)),
            "resolution": to_serializable(record.get("resolution")),
        }
    return record_map


def read_video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")

    try:
        return {
            "num_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def resolve_relative_video_path(relative_path: str, source_video_dir: Path | None) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        return relative

    candidate_paths: list[Path] = []
    if source_video_dir is not None:
        candidate_paths.append(source_video_dir.parent / relative)
        candidate_paths.append(source_video_dir / relative.name)
        if relative.parts and relative.parts[0] == source_video_dir.name:
            candidate_paths.append(source_video_dir / Path(*relative.parts[1:]))
        else:
            candidate_paths.append(source_video_dir / relative)

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    return candidate_paths[0] if candidate_paths else relative


def resolve_source_video_spec(
    sample_meta: dict[str, Any],
    source_video_dir: Path | None,
    filter_records: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    uttid = sample_meta.get("uttid")
    filter_record = filter_records.get(uttid)

    if filter_record is not None:
        video_path = resolve_relative_video_path(filter_record["relative_path"], source_video_dir)
        return {
            "video_path": video_path,
            "cut_start": filter_record["cut_start"],
            "cut_end": filter_record["cut_end"],
            "crop": filter_record.get("crop"),
            "source_stem": filter_record.get("source_stem"),
            "resolved_from": "filter_json",
        }

    source_video_stem = sample_meta.get("source_video_stem")
    cut_start = sample_meta.get("cut_start")
    cut_end = sample_meta.get("cut_end")
    if source_video_dir is None or source_video_stem is None or cut_start is None or cut_end is None:
        return None

    return {
        "video_path": source_video_dir / f"{source_video_stem}.mp4",
        "cut_start": cut_start,
        "cut_end": cut_end,
        "crop": None,
        "source_stem": source_video_stem,
        "resolved_from": "filename",
    }


def ensure_valid_crop(crop: list[int] | None, metadata: dict[str, Any]) -> list[int]:
    if crop is not None and len(crop) == 4:
        return [int(v) for v in crop]
    return [0, metadata["width"], 0, metadata["height"]]


def read_video_frames_by_indices(video_path: Path, frame_indices: list[int]) -> np.ndarray:
    if not frame_indices:
        return np.empty((0, 0, 0, 3), dtype=np.uint8)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open source video: {video_path}")

    frames = []
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = capture.read()
            if not ok:
                raise RuntimeError(f"Failed to read frame {frame_idx} from source video: {video_path}")
            frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    return np.stack(frames, axis=0)


def center_crop_to_aspect_ratio(video: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
    _, _, height, width = video.shape
    aspect_ratio_original = height / width
    aspect_ratio_target = target_height / target_width

    if aspect_ratio_original >= aspect_ratio_target:
        new_height = int(width * aspect_ratio_target)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width
    else:
        new_width = int(height / aspect_ratio_target)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height

    return video[:, :, top:bottom, left:right]


def preprocess_source_frames(
    frames: np.ndarray,
    target_height: int,
    target_width: int,
    crop: list[int],
) -> torch.Tensor:
    if frames.size == 0:
        return torch.empty((0, 3, target_height, target_width), dtype=torch.uint8)

    video = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    s_x, e_x, s_y, e_y = crop
    video = video[:, :, s_y:e_y, s_x:e_x]
    video = center_crop_to_aspect_ratio(video, target_height=target_height, target_width=target_width)
    video = TF.resize(video, (target_height, target_width))
    return video.round().clamp(0.0, 255.0).to(torch.uint8)


def pil_image_to_chw_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(rgb).permute(2, 0, 1)


def select_best_alignment_start(
    video_path: Path,
    cut_start: int,
    cut_end: int,
    target_num_frames: int,
    target_height: int,
    target_width: int,
    crop: list[int],
    first_frame_image: Image.Image | None,
) -> tuple[int, float | None]:
    if target_num_frames <= 0:
        raise ValueError("`target_num_frames` must be positive when aligning source clips.")

    max_start = cut_end - target_num_frames
    if max_start < cut_start:
        raise ValueError(
            f"Cannot align source clip because cut range [{cut_start}, {cut_end}) is shorter than "
            f"the requested decoded clip length ({target_num_frames})."
        )

    if first_frame_image is None:
        return cut_start, None

    candidate_starts = list(range(cut_start, max_start + 1))
    candidate_first_frames = read_video_frames_by_indices(video_path, candidate_starts)
    candidate_first_frames = preprocess_source_frames(
        candidate_first_frames,
        target_height=target_height,
        target_width=target_width,
        crop=crop,
    )

    reference_first_frame = pil_image_to_chw_tensor(first_frame_image)
    mse = (
        (candidate_first_frames.to(torch.float32) - reference_first_frame.unsqueeze(0).to(torch.float32)) ** 2
    ).mean(dim=(1, 2, 3))
    best_idx = int(mse.argmin().item())
    return candidate_starts[best_idx], float(mse[best_idx].item())


def load_reference_clip_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    target_height: int,
    target_width: int,
    crop: list[int],
) -> np.ndarray:
    frame_indices = list(range(start_frame, start_frame + num_frames))
    reference_frames = read_video_frames_by_indices(video_path, frame_indices)
    reference_frames = preprocess_source_frames(
        reference_frames,
        target_height=target_height,
        target_width=target_width,
        crop=crop,
    )
    return reference_frames.permute(0, 2, 3, 1).cpu().numpy()


def get_lpips_model(cache: dict[tuple[str, str], Any], device: str, net: str):
    cache_key = (device, net)
    if cache_key not in cache:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("LPIPS is not installed. Please install `lpips` before enabling metric computation.") from exc

        model = lpips.LPIPS(net=net).to(device)
        model.eval()
        cache[cache_key] = model
    return cache[cache_key]


def compute_lpips_score(
    predicted_frames: np.ndarray,
    reference_frames: np.ndarray,
    device: str,
    batch_size: int,
    model,
) -> float:
    if predicted_frames.shape != reference_frames.shape:
        raise ValueError(
            f"Predicted and reference videos must have the same shape, got "
            f"{predicted_frames.shape} vs {reference_frames.shape}"
        )

    total_score = 0.0
    total_frames = predicted_frames.shape[0]
    with torch.no_grad():
        for start_idx in range(0, total_frames, batch_size):
            end_idx = min(total_frames, start_idx + batch_size)
            pred_batch = torch.from_numpy(predicted_frames[start_idx:end_idx]).permute(0, 3, 1, 2).float()
            ref_batch = torch.from_numpy(reference_frames[start_idx:end_idx]).permute(0, 3, 1, 2).float()
            pred_batch = pred_batch.to(device) / 127.5 - 1.0
            ref_batch = ref_batch.to(device) / 127.5 - 1.0
            batch_scores = model(pred_batch, ref_batch)
            total_score += float(batch_scores.flatten().sum().item())
    return total_score / total_frames


def compute_compression_metrics(
    file_path: Path,
    vae_latent: torch.Tensor,
    sample_meta: dict[str, Any],
) -> dict[str, Any]:
    num_frame = sample_meta.get("num_frame")
    height = sample_meta.get("height")
    width = sample_meta.get("width")
    if num_frame is None or height is None or width is None:
        raise ValueError(f"Missing shape metadata in file name: {file_path.name}")

    total_pixels = int(num_frame) * int(height) * int(width)
    latent_payload_bytes = int(vae_latent.numel() * vae_latent.element_size())
    pt_container_bytes = int(file_path.stat().st_size)

    return {
        "num_frame": int(num_frame),
        "height": int(height),
        "width": int(width),
        "latent_payload_bytes": latent_payload_bytes,
        "latent_payload_bits": latent_payload_bytes * 8,
        "latent_payload_bpp": latent_payload_bytes * 8 / total_pixels,
        "pt_container_bytes": pt_container_bytes,
        "pt_container_bits": pt_container_bytes * 8,
        "pt_container_bpp": pt_container_bytes * 8 / total_pixels,
        "latent_dtype": str(vae_latent.dtype),
        "latent_shape": list(vae_latent.shape),
    }


def compute_sample_metrics(
    file_path: Path,
    sample_meta: dict[str, Any],
    payload: dict[str, Any],
    decoded_video_frames: np.ndarray | None,
    args: argparse.Namespace,
    source_video_dir: Path | None,
    filter_records: dict[str, dict[str, Any]],
    lpips_cache: dict[tuple[str, str], Any],
    device: str,
) -> dict[str, Any]:
    vae_latent = payload["vae_latent"]
    metrics = {
        "sample": file_path.stem,
        "file_path": str(file_path),
        "source_video_dir": str(source_video_dir) if source_video_dir is not None else None,
        "filter_json": args.filter_json,
    }
    metrics.update(compute_compression_metrics(file_path=file_path, vae_latent=vae_latent, sample_meta=sample_meta))

    if decoded_video_frames is None:
        metrics["lpips"] = None
        metrics["lpips_status"] = "skipped: decoded video unavailable"
        return metrics

    if source_video_dir is None:
        metrics["lpips"] = None
        metrics["lpips_status"] = "skipped: source video dir missing"
        return metrics

    source_spec = resolve_source_video_spec(
        sample_meta=sample_meta,
        source_video_dir=source_video_dir,
        filter_records=filter_records,
    )
    if source_spec is None:
        metrics["lpips"] = None
        metrics["lpips_status"] = "skipped: source metadata unresolved"
        return metrics

    source_video_path = Path(source_spec["video_path"])
    if not source_video_path.exists():
        metrics["lpips"] = None
        metrics["lpips_status"] = f"skipped: source video not found ({source_video_path})"
        return metrics

    try:
        source_metadata = read_video_metadata(source_video_path)
        crop = ensure_valid_crop(source_spec.get("crop"), source_metadata)
        alignment_start, first_frame_mse = select_best_alignment_start(
            video_path=source_video_path,
            cut_start=int(source_spec["cut_start"]),
            cut_end=int(source_spec["cut_end"]),
            target_num_frames=int(decoded_video_frames.shape[0]),
            target_height=int(decoded_video_frames.shape[1]),
            target_width=int(decoded_video_frames.shape[2]),
            crop=crop,
            first_frame_image=payload.get("first_frames_image"),
        )
        reference_frames = load_reference_clip_frames(
            video_path=source_video_path,
            start_frame=alignment_start,
            num_frames=int(decoded_video_frames.shape[0]),
            target_height=int(decoded_video_frames.shape[1]),
            target_width=int(decoded_video_frames.shape[2]),
            crop=crop,
        )
        lpips_model = get_lpips_model(cache=lpips_cache, device=device, net=args.lpips_net)
        lpips_score = compute_lpips_score(
            predicted_frames=decoded_video_frames,
            reference_frames=reference_frames,
            device=device,
            batch_size=args.metric_batch_size,
            model=lpips_model,
        )
        metrics.update(
            {
                "lpips": lpips_score,
                "lpips_status": "ok",
                "lpips_frames": int(decoded_video_frames.shape[0]),
                "alignment_start_frame": alignment_start,
                "alignment_first_frame_mse": first_frame_mse,
                "source_video_path": str(source_video_path),
                "source_video_resolved_from": source_spec.get("resolved_from"),
                "source_cut_start": int(source_spec["cut_start"]),
                "source_cut_end": int(source_spec["cut_end"]),
                "source_crop": crop,
            }
        )
    except Exception as exc:
        metrics["lpips"] = None
        metrics["lpips_status"] = f"failed: {exc}"

    return metrics


def summarize_metric_records(metric_records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "num_samples": len(metric_records),
        "num_samples_with_lpips": sum(record.get("lpips") is not None for record in metric_records),
        "averages": {},
        "samples": [],
    }

    numeric_metric_keys = [
        "latent_payload_bpp",
        "pt_container_bpp",
        "lpips",
        "alignment_first_frame_mse",
    ]
    for key in numeric_metric_keys:
        values = [float(record[key]) for record in metric_records if isinstance(record.get(key), (int, float))]
        if values:
            summary["averages"][key] = sum(values) / len(values)

    for record in metric_records:
        summary["samples"].append(
            {
                "sample": record["sample"],
                "latent_payload_bpp": record.get("latent_payload_bpp"),
                "pt_container_bpp": record.get("pt_container_bpp"),
                "lpips": record.get("lpips"),
                "lpips_status": record.get("lpips_status"),
            }
        )

    return summary


def load_vae(base_model_path: str, device: str):
    from diffusers.models import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(base_model_path, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device)
    vae.eval()
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=vae.dtype, device=device).view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std, dtype=vae.dtype, device=device).view(1, vae.config.z_dim, 1, 1, 1)
    )
    return vae, latents_mean, latents_std


def decode_latents_to_video(
    vae_latent: torch.Tensor,
    base_model_path: str,
    device: str,
    decode_sections: int | None,
) -> np.ndarray:
    if decode_sections == 0:
        raise ValueError("`decode_sections` must be > 0 when decoding is enabled.")

    sections = int(vae_latent.shape[0]) if decode_sections is None else min(decode_sections, int(vae_latent.shape[0]))
    latents = vae_latent[:sections]  
    vae, latents_mean, latents_std = load_vae(base_model_path, device)

    with torch.no_grad():
        normalized = latents.to(device=device, dtype=vae.dtype) / latents_std + latents_mean
        decoded = vae.decode(normalized, return_dict=False)[0]

    decoded = decoded.detach().cpu()
    decoded = ((decoded.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).to(torch.uint8)
    video = decoded.permute(0, 2, 3, 4, 1).reshape(-1, decoded.shape[3], decoded.shape[4], 3).numpy()

    del vae
    if device == "cuda":
        torch.cuda.empty_cache()

    return video


def save_video_preview(
    video_frames: np.ndarray,
    output_dir: Path,
    stem: str,
    fps: int,
    num_preview_frames: int,
    save_video: bool,
) -> None:
    if video_frames.size == 0:
        return

    frame_indices = evenly_spaced_indices(len(video_frames), num_preview_frames)
    preview_frames = [video_frames[idx] for idx in frame_indices]
    make_contact_sheet(preview_frames, output_dir / f"{stem}_decoded_grid.png", title="Decoded Video Preview")

    if not save_video:
        return

    mp4_path = output_dir / f"{stem}_decoded.mp4"
    gif_path = output_dir / f"{stem}_decoded.gif"

    try:
        imageio.mimwrite(mp4_path, list(video_frames), fps=fps, macro_block_size=1)
        print(f"[saved] decoded video: {mp4_path}")
    except Exception as exc:
        print(f"[warn] failed to write mp4 ({exc}), writing GIF instead: {gif_path}")
        imageio.mimsave(gif_path, list(video_frames), fps=fps)


def inspect_file(
    file_path: Path,
    output_root: Path,
    args: argparse.Namespace,
    device: str,
    source_video_dir: Path | None,
    filter_records: dict[str, dict[str, Any]],
    lpips_cache: dict[tuple[str, str], Any],
) -> dict[str, Any] | None:
    payload = torch.load(file_path, map_location="cpu", weights_only=False) # 加载的源pt文件
    sample_meta = parse_sample_name(file_path)
    sample_out_dir = output_root / file_path.stem
    sample_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "file_path": str(file_path),
        "parsed_from_filename": sample_meta,
        "keys": sorted(payload.keys()) if isinstance(payload, dict) else None,
        "payload": to_serializable(payload),
    }

    write_summary_json(summary, sample_out_dir / "summary.json")

    prompt_raw = payload.get("prompt_raw") if isinstance(payload, dict) else None
    save_prompt_text(prompt_raw, sample_out_dir / "prompt.txt")
    save_first_frame_image(payload.get("first_frames_image"), sample_out_dir / "first_frame.png")

    video_frames: np.ndarray | None = None
    sample_metrics: dict[str, Any] | None = None
    if isinstance(payload, dict) and "vae_latent" in payload:
        vae_latent = payload["vae_latent"]
        save_latent_heatmap_preview(vae_latent, sample_out_dir / "latent_heatmap_grid.png", args.num_preview_frames)

        if args.latent_rgb_preview:
            save_latent_rgb_preview(vae_latent, sample_out_dir / "latent_pca_rgb_grid.png", args.num_preview_frames)

        if args.base_model_path and args.decode_sections != 0:
            video_frames = decode_latents_to_video(
                vae_latent=vae_latent,
                base_model_path=args.base_model_path,
                device=device,
                decode_sections=args.decode_sections,
            )
            save_video_preview(
                video_frames=video_frames,
                output_dir=sample_out_dir,
                stem=file_path.stem,
                fps=args.fps,
                num_preview_frames=args.num_preview_frames,
                save_video=args.save_video,
            )
        elif args.save_video:
            print("[warn] `--save-video` was set but `--base-model-path` is missing, skipping video export.")

        if args.compute_metrics:
            sample_metrics = compute_sample_metrics(
                file_path=file_path,
                sample_meta=sample_meta,
                payload=payload,
                decoded_video_frames=video_frames,
                args=args,
                source_video_dir=source_video_dir,
                filter_records=filter_records,
                lpips_cache=lpips_cache,
                device=device,
            )
            write_summary_json(sample_metrics, sample_out_dir / "metrics.json")

    prompt_preview = ""
    if isinstance(prompt_raw, str):
        prompt_preview = prompt_raw[: args.print_prompt_chars].replace("\n", " ")

    print(f"[done] {file_path}")
    print(f"  output_dir: {sample_out_dir}")
    print(f"  parsed: {sample_meta}")
    if prompt_preview:
        print(f"  prompt: {prompt_preview}")
    if isinstance(payload, dict) and "vae_latent" in payload:
        print(f"  vae_latent: {tuple(payload['vae_latent'].shape)} {payload['vae_latent'].dtype}")
    if isinstance(payload, dict) and "prompt_embed" in payload:
        print(f"  prompt_embed: {tuple(payload['prompt_embed'].shape)} {payload['prompt_embed'].dtype}")
    if sample_metrics is not None:
        print(f"  latent_payload_bpp: {sample_metrics['latent_payload_bpp']:.6f}")
        print(f"  pt_container_bpp: {sample_metrics['pt_container_bpp']:.6f}")
        if sample_metrics.get("lpips") is not None:
            print(f"  lpips: {sample_metrics['lpips']:.6f}")
        else:
            print(f"  lpips: {sample_metrics.get('lpips_status')}")

    return sample_metrics


def write_dataset_index(pt_files: list[Path], output_root: Path) -> None:
    parsed_samples = [parse_sample_name(file_path) for file_path in pt_files]
    bucket_counts: dict[str, int] = {}
    for sample in parsed_samples:
        bucket_key = f"{sample['num_frame']}x{sample['height']}x{sample['width']}"
        bucket_counts[bucket_key] = bucket_counts.get(bucket_key, 0) + 1

    dataset_index = {
        "num_files_considered": len(pt_files),
        "files": [str(file_path) for file_path in pt_files],
        "bucket_counts": bucket_counts,
        "samples": parsed_samples,
    }
    write_summary_json(dataset_index, output_root / "dataset_index.json")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    source_video_dir = Path(args.source_video_dir) if args.source_video_dir else None
    filter_records = load_filter_records(args.filter_json)
    lpips_cache: dict[tuple[str, str], Any] = {}

    if args.base_model_path is None and args.decode_sections != 0 and args.save_video:
        print(f"[info] no `--base-model-path` provided, latent decoding will be skipped.")

    pt_files = resolve_pt_files(input_path, args.max_files)
    write_dataset_index(pt_files, output_root)

    print(f"[info] input_path={input_path}")
    print(f"[info] files_to_process={len(pt_files)}")
    print(f"[info] output_dir={output_root}")
    print(f"[info] device={device}")
    if args.base_model_path:
        print(f"[info] base_model_path={args.base_model_path}")
    else:
        print("[info] base_model_path=None (metadata-only mode)")
    if source_video_dir is not None:
        print(f"[info] source_video_dir={source_video_dir}")
    if args.filter_json:
        print(f"[info] filter_json={args.filter_json}")

    metric_records: list[dict[str, Any]] = []
    for file_path in pt_files:
        sample_metrics = inspect_file(
            file_path=file_path,
            output_root=output_root,
            args=args,
            device=device,
            source_video_dir=source_video_dir,
            filter_records=filter_records,
            lpips_cache=lpips_cache,
        )
        if sample_metrics is not None:
            metric_records.append(sample_metrics)

    if metric_records:
        metrics_summary = summarize_metric_records(metric_records)
        write_summary_json(metrics_summary, output_root / "metrics_summary.json")
        print(f"[info] metrics summary saved to: {output_root / 'metrics_summary.json'}")
        if "latent_payload_bpp" in metrics_summary["averages"]:
            print(f"[info] avg latent_payload_bpp={metrics_summary['averages']['latent_payload_bpp']:.6f}")
        if "pt_container_bpp" in metrics_summary["averages"]:
            print(f"[info] avg pt_container_bpp={metrics_summary['averages']['pt_container_bpp']:.6f}")
        if "lpips" in metrics_summary["averages"]:
            print(f"[info] avg lpips={metrics_summary['averages']['lpips']:.6f}")

    print("[info] training reads the same fields from `helios/dataset/dataloader_history_latents_dist.py`:")
    print("       `vae_latent`, `prompt_embed`, `prompt_raw` (optional), `first_frames_image` (optional).")


if __name__ == "__main__":
    main()
