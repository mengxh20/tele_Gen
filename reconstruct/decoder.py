import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor
from tqdm import tqdm


ROOT_DIR = "reconstruct/outputs_CNN"
DEFAULT_INPUT_DIR = f"{ROOT_DIR}/recover_latents"
DEFAULT_OUTPUT_DIR = f"{ROOT_DIR}/Videos"

LATENT_FORMAT_V1 = "helios_vae_latent_v1"
LATENT_FORMAT_V2 = "helios_vae_latent_v2"
ModelBundle = Tuple[AutoencoderKLWan, VideoProcessor, torch.Tensor, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode normalized Helios VAE latents back into videos.")
    parser.add_argument("--root_dir",'-R', type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--input_dir",'-I', type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output_dir",'-O', type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base_model_path", type=str, default="/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base")
    parser.add_argument("--device", type=str, default="cuda")
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


def decode_payload_to_output_path(
    payload: Dict,
    latent_path: Path,
    output_path: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(
        list(reconstructed_video),
        output_video_path=str(output_path),
        fps=payload["source_fps"],
        macro_block_size=1,
    )
    print(
        f"[decoder] saved={output_path} "
        f"frames={reconstructed_video.shape[0]} "
        f"resolution={reconstructed_video.shape[2]}x{reconstructed_video.shape[1]}"
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output_path


def decode_latent_file_to_output_path(
    latent_path: Path,
    output_path: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
) -> Path:
    payload = load_payload(latent_path)
    return decode_payload_to_output_path(
        payload=payload,
        latent_path=latent_path,
        output_path=output_path,
        cli_model_path=cli_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
    )


def decode_latent_file(
    latent_path: Path,
    input_dir: Path,
    output_dir: Path,
    cli_model_path: str,
    model_bundle_cache: Dict[str, ModelBundle],
    device: torch.device,
) -> Path:
    relative_path = latent_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix(".mp4")
    return decode_latent_file_to_output_path(
        latent_path=latent_path,
        output_path=output_path,
        cli_model_path=cli_model_path,
        model_bundle_cache=model_bundle_cache,
        device=device,
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    latent_paths = discover_latent_paths(input_dir)

    print(f"[decoder] input_dir={input_dir}")
    print(f"[decoder] output_dir={output_dir}")
    print(f"[decoder] base_model_path={args.base_model_path or 'from-latent-metadata'}")

    model_bundle_cache: Dict[str, ModelBundle] = {}
    for latent_path in tqdm(latent_paths, desc="Decoding latents"):
        decode_latent_file(
            latent_path=latent_path,
            input_dir=input_dir,
            output_dir=output_dir,
            cli_model_path=args.base_model_path,
            model_bundle_cache=model_bundle_cache,
            device=device,
        )


if __name__ == "__main__":
    main()
