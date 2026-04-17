import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruct.codec_gop import make_section_ranges


ROOT_DIR = "reconstruct/gen2recon_runs_CNN_3/recover_outputs" # 只需要传一个 -R 参数
DEFAULT_INPUT_SUBDIR = "recover_latents"
DEFAULT_OUTPUT_SUBDIR = "Videos"
DEFAULT_SOURCE_VIDEO_DIR = "reconstruct/videos"

LATENT_FORMAT_V1 = "helios_vae_latent_v1"
LATENT_FORMAT_V2 = "helios_vae_latent_v2"
COMPARE_MODE_CHOICES = ("off", "full", "sections", "both")
PIL_RESAMPLING_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
ModelBundle = Tuple[AutoencoderKLWan, VideoProcessor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class CompareExportConfig:
    source_video_dir: Path
    compare_output_dir: Path
    metrics_dir: Path
    compare_mode: str
    source_video_index: Dict[str, List[Path]]

    @property
    def export_full(self) -> bool:
        return self.compare_mode in {"full", "both"}

    @property
    def export_sections(self) -> bool:
        return self.compare_mode in {"sections", "both"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode normalized Helios VAE latents back into videos.")
    parser.add_argument("--root_dir", "-R", type=Path, default=Path(ROOT_DIR))
    parser.add_argument(
        "--input_dir",
        "-I",
        type=Path,
        default=None,
        help=f"Default: <root_dir>/{DEFAULT_INPUT_SUBDIR}",
    )
    parser.add_argument(
        "--output_dir",
        "-O",
        type=Path,
        default=None,
        help=f"Default: <root_dir>/{DEFAULT_OUTPUT_SUBDIR}",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/data/gemini/gemini-sharedata/platform/public/luojx/team/mengxh/MODELS/Helios-Base",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--source_video_dir", type=Path, default=Path(DEFAULT_SOURCE_VIDEO_DIR))
    parser.add_argument("--compare_output_dir", type=Path, default=None)
    parser.add_argument("--compare_mode", type=str, default="both", choices=COMPARE_MODE_CHOICES)
    return parser.parse_args()


def discover_latent_paths(input_dir: Path) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    latent_paths = [path for path in sorted(input_dir.rglob("*.pt")) if path.is_file()]
    if not latent_paths:
        raise FileNotFoundError(f"No latent files were found in: {input_dir}")
    return latent_paths


def resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.index is not None:
            torch.cuda.set_device(device.index)
    return device


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
    if not isinstance(payload["latent_chunks"], list) or not payload["latent_chunks"]:
        raise ValueError(f"Latent file {latent_path} does not contain any latent chunks.")
    if format_version == LATENT_FORMAT_V1 and len(payload["latent_chunks"]) != len(payload["chunk_frame_ranges"]):
        raise ValueError(
            f"Latent file {latent_path} has mismatched chunk metadata: "
            f"{len(payload['latent_chunks'])} latent chunks vs {len(payload['chunk_frame_ranges'])} frame ranges."
        )


def load_vae(base_model_path: str, device: torch.device) -> ModelBundle:
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


def load_payload(latent_path: Path) -> Dict:
    try:
        return torch.load(latent_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(latent_path, map_location="cpu")


def resolve_model_bundle(
    payload: Dict,
    latent_path: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
) -> ModelBundle:
    model_path = cli_model_path or payload.get("model_path")
    if not model_path:
        raise ValueError(
            f"Latent file {latent_path} does not include a usable model_path. "
            "Please pass --base_model_path explicitly."
        )

    if model_path not in model_bundle_cache:
        model_bundle_cache[model_path] = load_vae(model_path, device)
    return model_bundle_cache[model_path]


def save_video_frames(frames: np.ndarray, output_path: Path, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_array = to_uint8_frames(frames)

    export_to_video(
        [Image.fromarray(frame) for frame in frames_array],
        output_video_path=str(output_path),
        fps=fps,
        macro_block_size=1,
    )


def to_uint8_frames(frames: np.ndarray) -> np.ndarray:
    frames_array = np.asarray(frames)
    if frames_array.dtype == np.uint8:
        return frames_array
    if np.issubdtype(frames_array.dtype, np.floating):
        min_value = float(frames_array.min()) if frames_array.size > 0 else 0.0
        max_value = float(frames_array.max()) if frames_array.size > 0 else 1.0
        if min_value >= -1e-6 and max_value <= 1.0 + 1e-6:
            frames_array = np.clip(frames_array, 0.0, 1.0)
            return (frames_array * 255.0).round().astype(np.uint8)
        frames_array = np.clip(frames_array, 0.0, 255.0)
        return frames_array.round().astype(np.uint8)
    frames_array = np.clip(frames_array, 0, 255)
    return frames_array.astype(np.uint8)


def build_source_video_index(source_video_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not source_video_dir.exists():
        return index
    for video_path in sorted(source_video_dir.rglob("*.mp4")):
        index.setdefault(video_path.stem, []).append(video_path.resolve())
    return index


def build_compare_export_config(args: argparse.Namespace, root_dir: Path) -> Optional[CompareExportConfig]:
    if args.compare_mode == "off":
        return None
    source_video_dir = args.source_video_dir.resolve()
    compare_output_dir = (args.compare_output_dir.resolve() if args.compare_output_dir else (root_dir / "duibi_Videos").resolve())
    metrics_dir = (root_dir / "metrics").resolve()
    if not source_video_dir.exists():
        print(f"[decoder] warning: source_video_dir does not exist, compare export will be skipped: {source_video_dir}")
    return CompareExportConfig(
        source_video_dir=source_video_dir,
        compare_output_dir=compare_output_dir,
        metrics_dir=metrics_dir,
        compare_mode=args.compare_mode,
        source_video_index=build_source_video_index(source_video_dir),
    )


def resolve_source_video_path(relative_path: Path, compare_config: CompareExportConfig) -> Optional[Path]:
    relative_candidate = (compare_config.source_video_dir / relative_path).with_suffix(".mp4")
    if relative_candidate.exists():
        return relative_candidate.resolve()

    stem_matches = compare_config.source_video_index.get(relative_path.stem, [])
    if not stem_matches:
        return None
    if len(stem_matches) > 1:
        print(
            f"[decoder] warning: found multiple source videos for stem={relative_path.stem}, "
            f"using {stem_matches[0]}"
        )
    return stem_matches[0]


def load_video_frames(video_path: Path) -> np.ndarray:
    frames: List[np.ndarray] = []
    with imageio.get_reader(str(video_path)) as reader:
        for frame in reader:
            frames.append(np.asarray(frame, dtype=np.uint8))
    if not frames:
        raise RuntimeError(f"Video file does not contain readable frames: {video_path}")
    return np.stack(frames, axis=0)


def resize_video_frames(frames: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if frames.shape[1] == target_height and frames.shape[2] == target_width:
        return frames
    resized_frames = [
        np.asarray(Image.fromarray(frame).resize((target_width, target_height), PIL_RESAMPLING_LANCZOS), dtype=np.uint8)
        for frame in frames
    ]
    return np.stack(resized_frames, axis=0)


def align_compare_frames(
    source_frames: np.ndarray,
    reconstructed_frames: np.ndarray,
    expected_num_frames: Optional[int],
    latent_path: Path,
    source_video_path: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    max_frames = min(source_frames.shape[0], reconstructed_frames.shape[0])
    if expected_num_frames is not None:
        max_frames = min(max_frames, int(expected_num_frames))

    if source_frames.shape[0] != reconstructed_frames.shape[0]:
        print(
            f"[decoder] warning: frame count mismatch for compare export "
            f"(source={source_frames.shape[0]}, recover={reconstructed_frames.shape[0]}). "
            f"Using first {max_frames} frames. "
            f"source={source_video_path} latent={latent_path}"
        )
    if max_frames <= 0:
        raise RuntimeError(
            f"Unable to build compare video because no aligned frames are available. "
            f"source={source_video_path} latent={latent_path}"
        )

    return source_frames[:max_frames], reconstructed_frames[:max_frames]


def build_compare_frames(source_frames: np.ndarray, reconstructed_frames: np.ndarray) -> np.ndarray:
    source_frames = to_uint8_frames(source_frames)
    reconstructed_frames = to_uint8_frames(reconstructed_frames)
    reconstructed_frames = resize_video_frames(
        reconstructed_frames,
        target_height=source_frames.shape[1],
        target_width=source_frames.shape[2],
    )
    return np.concatenate([source_frames, reconstructed_frames], axis=2)


def resolve_section_ranges(payload: Dict, latent_path: Path, metrics_path: Path) -> List[Tuple[int, int]]:
    recovery_metadata = payload.get("recovery_metadata") or {}
    section_ranges = recovery_metadata.get("section_ranges")
    if section_ranges is not None:
        return [tuple(int(value) for value in section_range) for section_range in section_ranges]

    if not metrics_path.exists():
        print(
            f"[decoder] warning: section_ranges missing in payload and metrics file not found, "
            f"skip section compare export for {latent_path}"
        )
        return []

    with metrics_path.open("r", encoding="utf-8") as fp:
        metrics = json.load(fp)
    codec_config = metrics.get("codec_config") or {}
    section_span_latents = codec_config.get("section_span_latents")
    if section_span_latents is None:
        print(
            f"[decoder] warning: section_span_latents missing in metrics, "
            f"skip section compare export for {latent_path}"
        )
        return []

    total_latent_frames = sum(int(chunk.shape[1]) for chunk in payload["latent_chunks"])
    return make_section_ranges(
        total_latent_frames=total_latent_frames,
        section_span_latents=int(section_span_latents),
        start_index=1,
    )


def section_latents_to_frame_range(section_range: Tuple[int, int], source_num_frames: int) -> Optional[Tuple[int, int]]:
    section_start, section_end = (int(section_range[0]), int(section_range[1]))
    frame_start = 1 + 4 * (section_start - 1)
    frame_end = min(int(source_num_frames), 1 + 4 * (section_end - 1))
    if frame_end <= frame_start:
        return None
    return frame_start, frame_end


def export_compare_outputs(
    payload: Dict,
    latent_path: Path,
    relative_path: Path,
    reconstructed_video: np.ndarray,
    compare_config: Optional[CompareExportConfig],
) -> None:
    if compare_config is None:
        return

    source_video_path = resolve_source_video_path(relative_path, compare_config)
    if source_video_path is None:
        print(
            f"[decoder] warning: source video not found for compare export, skip. "
            f"latent={latent_path} source_root={compare_config.source_video_dir}"
        )
        return

    source_frames = load_video_frames(source_video_path)
    expected_num_frames = payload.get("source_num_frames")
    source_frames, reconstructed_frames = align_compare_frames(
        source_frames=source_frames,
        reconstructed_frames=reconstructed_video,
        expected_num_frames=expected_num_frames,
        latent_path=latent_path,
        source_video_path=source_video_path,
    )
    compare_frames = build_compare_frames(source_frames, reconstructed_frames)
    compare_fps = float(payload["source_fps"])

    if compare_config.export_full:
        compare_output_path = compare_config.compare_output_dir / relative_path.with_suffix(".mp4")
        save_video_frames(compare_frames, compare_output_path, fps=compare_fps)
        print(
            f"[decoder] saved_compare={compare_output_path} "
            f"frames={compare_frames.shape[0]} "
            f"resolution={compare_frames.shape[2]}x{compare_frames.shape[1]}"
        )

    if not compare_config.export_sections:
        return

    metrics_path = compare_config.metrics_dir / relative_path.with_suffix(".json")
    section_ranges = resolve_section_ranges(payload=payload, latent_path=latent_path, metrics_path=metrics_path)
    if not section_ranges:
        return

    available_frames = compare_frames.shape[0]
    section_output_dir = compare_config.compare_output_dir / relative_path.with_suffix("")
    saved_section_count = 0
    for section_idx, section_range in enumerate(section_ranges):
        frame_range = section_latents_to_frame_range(section_range, source_num_frames=available_frames)
        if frame_range is None:
            continue
        frame_start, frame_end = frame_range
        section_frames = compare_frames[frame_start:frame_end]
        if section_frames.shape[0] == 0:
            continue
        section_output_path = section_output_dir / (
            f"section_{section_idx:03d}_f{frame_start:04d}-f{frame_end - 1:04d}.mp4"
        )
        save_video_frames(section_frames, section_output_path, fps=compare_fps)
        saved_section_count += 1

    print(
        f"[decoder] saved_section_compare_dir={section_output_dir} "
        f"sections={saved_section_count}"
    )


def decode_payload_to_output_path(
    payload: Dict,
    latent_path: Path,
    relative_path: Path,
    output_path: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
    compare_config: Optional[CompareExportConfig],
) -> Path:
    validate_payload(payload, latent_path)
    format_version = payload["format_version"]
    vae, video_processor, latents_mean, latents_std = resolve_model_bundle(
        payload=payload,
        latent_path=latent_path,
        cli_model_path=cli_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
    )

    frame_chunks: List[np.ndarray] = []
    with torch.inference_mode():
        chunk_frame_ranges = payload.get("chunk_frame_ranges")
        for chunk_idx, latent_chunk in enumerate(payload["latent_chunks"]):
            latent_chunk = latent_chunk.unsqueeze(0).to(device=device, dtype=vae.dtype)
            decoded = vae.decode(latent_chunk / latents_std + latents_mean, return_dict=False)[0]
            decoded_frames = video_processor.postprocess_video(decoded, output_type="np")[0]

            if chunk_frame_ranges is not None:
                frame_range = chunk_frame_ranges[chunk_idx]
                expected_frames = frame_range[1] - frame_range[0]
                if decoded_frames.shape[0] != expected_frames:
                    raise RuntimeError(
                        f"Decoded frame count mismatch for {latent_path}. "
                        f"Expected {expected_frames}, got {decoded_frames.shape[0]}."
                    )
            elif format_version == LATENT_FORMAT_V1:
                raise RuntimeError(
                    f"Latent file {latent_path} is missing chunk_frame_ranges for {LATENT_FORMAT_V1}."
                )
            frame_chunks.append(decoded_frames)

    reconstructed_video = np.concatenate(frame_chunks, axis=0)
    source_num_frames = payload.get("source_num_frames")
    if source_num_frames is not None:
        if reconstructed_video.shape[0] < source_num_frames:
            raise RuntimeError(
                f"Decoded frame count mismatch for {latent_path}. "
                f"Expected at least {source_num_frames}, got {reconstructed_video.shape[0]}."
            )
        reconstructed_video = reconstructed_video[:source_num_frames]

    save_video_frames(reconstructed_video, output_path, fps=float(payload["source_fps"]))
    print(
        f"[decoder] saved={output_path} "
        f"frames={reconstructed_video.shape[0]} "
        f"resolution={reconstructed_video.shape[2]}x{reconstructed_video.shape[1]}"
    )
    export_compare_outputs(
        payload=payload,
        latent_path=latent_path,
        relative_path=relative_path,
        reconstructed_video=reconstructed_video,
        compare_config=compare_config,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_path


def decode_latent_file_to_output_path(
    latent_path: Path,
    relative_path: Path,
    output_path: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
    compare_config: Optional[CompareExportConfig],
) -> Path:
    payload = load_payload(latent_path)
    return decode_payload_to_output_path(
        payload=payload,
        latent_path=latent_path,
        relative_path=relative_path,
        output_path=output_path,
        cli_model_path=cli_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
        compare_config=compare_config,
    )


def decode_latent_file(
    latent_path: Path,
    input_dir: Path,
    output_dir: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
    compare_config: Optional[CompareExportConfig],
) -> Path:
    relative_path = latent_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".mp4")
    return decode_latent_file_to_output_path(
        latent_path=latent_path,
        relative_path=relative_path,
        output_path=output_path,
        cli_model_path=cli_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
        compare_config=compare_config,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    root_dir = args.root_dir.resolve()
    input_dir = args.input_dir.resolve() if args.input_dir else (root_dir / DEFAULT_INPUT_SUBDIR).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (root_dir / DEFAULT_OUTPUT_SUBDIR).resolve()
    compare_config = build_compare_export_config(args, root_dir)
    latent_paths = discover_latent_paths(input_dir)

    print(f"[decoder] input_dir={input_dir}")
    print(f"[decoder] output_dir={output_dir}")
    print(f"[decoder] base_model_path={args.base_model_path or 'from-latent-metadata'}")
    if compare_config is None:
        print("[decoder] compare_mode=off")
    else:
        print(f"[decoder] source_video_dir={compare_config.source_video_dir}")
        print(f"[decoder] compare_output_dir={compare_config.compare_output_dir}")
        print(f"[decoder] compare_mode={compare_config.compare_mode}")

    model_bundle_cache: Dict[str, ModelBundle] = {}
    for latent_path in tqdm(latent_paths, desc="Decoding latents"):
        decode_latent_file(
            latent_path=latent_path,
            input_dir=input_dir,
            output_dir=output_dir,
            cli_model_path=args.base_model_path,
            model_bundle_cache=model_bundle_cache,
            device=device,
            compare_config=compare_config,
        )


if __name__ == "__main__":
    main()
