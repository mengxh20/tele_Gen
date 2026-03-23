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

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
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
        return {"uttid": file_path.stem, "num_frame": None, "height": None, "width": None}

    try:
        uttid = "_".join(parts[:-3])
        num_frame = int(parts[-3])
        height = int(parts[-2])
        width = int(parts[-1])
        return {
            "uttid": uttid,
            "num_frame": num_frame,
            "height": height,
            "width": width,
        }
    except ValueError:
        return {"uttid": file_path.stem, "num_frame": None, "height": None, "width": None}


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
) -> None:
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

    for file_path in pt_files:
        inspect_file(
            file_path=file_path,
            output_root=output_root,
            args=args,
            device=device,
        )

    print("[info] training reads the same fields from `helios/dataset/dataloader_history_latents_dist.py`:")
    print("       `vae_latent`, `prompt_embed`, `prompt_raw` (optional), `first_frames_image` (optional).")


if __name__ == "__main__":
    main()
