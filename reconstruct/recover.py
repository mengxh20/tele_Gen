import argparse
import json
import math
import os
import random
import sys
import types
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from reconstruct.latent_io import (
    DEFAULT_BASE_MODEL_PATH,
    LATENT_FORMAT_V2,
    LOW_LATENT_FORMAT_V1,
    build_chunk_frame_ranges,
    compute_bpp_from_total_pixels,
    flatten_latent_chunks,
    infer_source_num_frames,
    load_payload,
    resolve_device,
    resolve_source_resolution,
    split_full_latents,
    validate_payload,
)


DEFAULT_INPUT_PATH = Path("reconstruct/latents")
DEFAULT_TRAIN_OUTPUT_DIR = Path("reconstruct/gen2recon_runs")
DEFAULT_INFER_OUTPUT_DIR = Path("reconstruct/recover_outputs")
DEFAULT_TEMPORAL_FACTOR = 2
DEFAULT_SPATIAL_FACTOR = 2
DEFAULT_QUANT_DTYPE = "int8"
DEFAULT_KEYFRAME_DTYPE = "float16"
DEFAULT_HISTORY_SIZES = [16, 2, 1]
DEFAULT_LATENT_WINDOW_SIZE = 9
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_WEIGHT_DTYPE = "bf16"
DEFAULT_LEARNING_RATE = 1e-5
RECOVER_CONFIG_VERSION = "helios_recover_v1"
EPS = 1e-8


@dataclass
class CodecConfig:
    temporal_factor: int = DEFAULT_TEMPORAL_FACTOR
    spatial_factor: int = DEFAULT_SPATIAL_FACTOR
    quant_dtype: str = DEFAULT_QUANT_DTYPE
    keyframe_dtype: str = DEFAULT_KEYFRAME_DTYPE


@dataclass
class SequenceMetadata:
    source_fps: float
    source_num_frames: int
    source_width: int
    source_height: int
    total_pixels: int
    raw_file_bytes: int
    raw_bpp: float
    chunk_lengths: List[int]
    chunk_frame_ranges: List[Tuple[int, int]]


@dataclass
class PreparedSequence:
    path: Path
    metadata: SequenceMetadata
    clean_full_latents: torch.Tensor
    low_codec_payload: Dict[str, object]
    low_full_latents: torch.Tensor


@dataclass(frozen=True)
class DistributedContext:
    is_distributed: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


class LatentWindowDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[PreparedSequence],
        history_sizes: Sequence[int],
        latent_window_size: int,
    ):
        self.sequences = list(sequences)
        self.history_sizes = list(history_sizes)
        self.history_window_size = sum(history_sizes)
        self.latent_window_size = latent_window_size
        self.samples: List[Tuple[int, int]] = []

        for seq_idx, sequence in enumerate(self.sequences):
            total_latent_frames = sequence.clean_full_latents.shape[1]
            for start in range(1, total_latent_frames, latent_window_size):
                self.samples.append((seq_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | int]:
        seq_idx, section_start = self.samples[index]
        sequence = self.sequences[seq_idx]

        target_latents, valid_target_frames = extract_target_window(
            clean_full_latents=sequence.clean_full_latents,
            section_start=section_start,
            latent_window_size=self.latent_window_size,
        )
        history_latents = extract_history_window(
            low_full_latents=sequence.low_full_latents,
            section_start=section_start,
            history_window_size=self.history_window_size,
        )

        return {
            "history_latents": history_latents,
            "target_latents": target_latents,
            "x0_latents": sequence.clean_full_latents[:, :1],
            "valid_target_frames": valid_target_frames,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run Helios latent recovery for generative compression.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the latent recovery network from ori_latents.")
    add_common_train_args(train_parser)

    infer_parser = subparsers.add_parser("infer", help="Compress ori_latents and reconstruct recover_latents.")
    add_common_infer_args(infer_parser)

    return parser.parse_args()


def add_common_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--weight_dtype", type=str, default=DEFAULT_WEIGHT_DTYPE, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal_factor", type=int, default=DEFAULT_TEMPORAL_FACTOR)
    parser.add_argument("--spatial_factor", type=int, default=DEFAULT_SPATIAL_FACTOR)
    parser.add_argument("--quant_dtype", type=str, default=DEFAULT_QUANT_DTYPE, choices=["int8"])
    parser.add_argument("--keyframe_dtype", type=str, default=DEFAULT_KEYFRAME_DTYPE, choices=["float16", "float32"])
    parser.add_argument("--history_sizes", type=int, nargs="+", default=DEFAULT_HISTORY_SIZES)
    parser.add_argument("--latent_window_size", type=int, default=DEFAULT_LATENT_WINDOW_SIZE)


def add_common_infer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input_path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_INFER_OUTPUT_DIR)
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
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)


def main() -> None:
    args = parse_args()
    if args.command == "train":
        command_train(args)
    elif args.command == "infer":
        command_infer(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def ensure_diffusers_parallel_shim() -> None:
    try:
        import diffusers.models._modeling_parallel  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    shim_module = types.ModuleType("diffusers.models._modeling_parallel")

    class ContextParallelInput:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class ContextParallelOutput(ContextParallelInput):
        pass

    shim_module.ContextParallelInput = ContextParallelInput
    shim_module.ContextParallelOutput = ContextParallelOutput
    sys.modules["diffusers.models._modeling_parallel"] = shim_module


def import_helios_inference_stack():
    ensure_diffusers_parallel_shim()
    try:
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
        from helios.utils.utils_base import encode_prompt
    except ImportError as exc:
        raise ImportError(
            "Failed to import the Helios transformer stack. "
            "Please use the project's intended diffusers environment before running recover.py."
        ) from exc

    return HeliosTransformer3DModel, HeliosScheduler, encode_prompt


def import_stage1_prepare_fn():
    from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents

    return prepare_stage1_clean_input_from_latents


def parse_accelerator_mixed_precision(weight_dtype: str) -> str:
    if weight_dtype == "bf16":
        return "bf16"
    if weight_dtype == "fp16":
        return "fp16"
    if weight_dtype == "fp32":
        return "no"
    raise ValueError(f"Unsupported mixed precision weight dtype: {weight_dtype}")


def create_train_accelerator(weight_dtype: str, gradient_accumulation_steps: int):
    try:
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs
    except ImportError as exc:
        raise ImportError(
            "Recover training requires accelerate. Please activate the project's intended training environment."
        ) from exc

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_kwargs = InitProcessGroupKwargs(backend=backend, timeout=timedelta(seconds=1800))
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=parse_accelerator_mixed_precision(weight_dtype),
        kwargs_handlers=[init_kwargs],
    )


def resolve_train_device(device_arg: str) -> torch.device:
    requested_device = torch.device(device_arg)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if requested_device.type != "cuda" or world_size <= 1:
        return resolve_device(device_arg)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is not available.")

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("Accelerate training requires LOCAL_RANK to be set by the launcher.")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


def command_train(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("RANK", "0"))
    device = resolve_train_device(args.device)

    set_seed(args.seed + rank)
    weight_dtype = parse_weight_dtype(args.weight_dtype)
    codec_config = build_codec_config(args)
    history_sizes = normalize_history_sizes(args.history_sizes)
    input_root, latent_paths = discover_input_latent_paths(args.input_path)
    prepared_sequences = [prepare_sequence(path, codec_config) for path in latent_paths]
    dataset = LatentWindowDataset(
        sequences=prepared_sequences,
        history_sizes=history_sizes,
        latent_window_size=args.latent_window_size,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training windows were built from the provided latent files.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )

    transformer, _scheduler, prompt_embeds = load_transformer_bundle(
        base_model_path=args.base_model_path,
        device=device,
        weight_dtype=weight_dtype,
        checkpoint_dir=None,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    transformer.train()
    base_transformer = transformer
    trainable_params = count_trainable_parameters(base_transformer)

    optimizer = torch.optim.AdamW(
        [param for param in transformer.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    accelerator = create_train_accelerator(
        weight_dtype=args.weight_dtype,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    if accelerator.state.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.batch_size

    transformer, optimizer, dataloader = accelerator.prepare(transformer, optimizer, dataloader)

    output_dir = args.output_dir.resolve()
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[train] input_root={input_root} files={len(latent_paths)} windows={len(dataset)} "
            f"device={device} world_size={accelerator.num_processes} "
            f"effective_global_batch_size={args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
        )
    accelerator.wait_for_everyone()

    train_metrics: Dict[str, object] = {
        "checkpoint_format_version": RECOVER_CONFIG_VERSION,
        "losses": [],
        "steps": 0,
        "trainable_params": trainable_params,
        "world_size": accelerator.num_processes,
        "per_device_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * accelerator.num_processes,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "input_path": str(args.input_path.resolve()),
        "num_sequences": len(prepared_sequences),
    }

    progress = None
    if args.max_steps is not None and accelerator.is_main_process:
        progress = tqdm(total=args.max_steps, desc="Training")

    global_step = 0
    stop_training = False
    running_loss: Optional[torch.Tensor] = None
    running_micro_steps = 0
    for epoch in range(args.epochs):
        epoch_iterator = tqdm(
            dataloader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=not accelerator.is_main_process,
        )
        for batch in epoch_iterator:
            with accelerator.accumulate(transformer):
                loss = training_step(
                    batch=batch,
                    transformer=transformer,
                    prompt_embeds=prompt_embeds,
                    device=device,
                    weight_dtype=weight_dtype,
                    history_sizes=history_sizes,
                    latent_window_size=args.latent_window_size,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            loss_for_logging = loss.detach().float()
            running_loss = loss_for_logging if running_loss is None else running_loss + loss_for_logging
            running_micro_steps += 1

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            reduced_loss = accelerator.reduce(running_loss / running_micro_steps, reduction="mean")
            running_loss = None
            running_micro_steps = 0
            if accelerator.is_main_process:
                train_metrics["steps"] = global_step
                train_metrics["losses"].append(float(reduced_loss.cpu().item()))
                if global_step % args.log_every == 0:
                    epoch_iterator.set_postfix(loss=f"{train_metrics['losses'][-1]:.6f}")
                if progress is not None:
                    progress.update(1)

            if args.max_steps is not None and global_step >= args.max_steps:
                stop_training = True
                break

        if stop_training:
            break

    if progress is not None:
        progress.close()

    save_training_artifacts(
        output_dir=output_dir,
        transformer=transformer,
        codec_config=codec_config,
        history_sizes=history_sizes,
        latent_window_size=args.latent_window_size,
        args=args,
        train_metrics=train_metrics,
        accelerator=accelerator,
    )
    if accelerator.is_main_process and train_metrics["losses"]:
        print(
            f"[train] completed steps={train_metrics['steps']} "
            f"last_loss={train_metrics['losses'][-1]:.6f} "
            f"trainable_params={train_metrics['trainable_params']}"
        )


def command_infer(args: argparse.Namespace) -> None:
    distributed_context, device = init_distributed_context(args.device)
    try:
        set_seed(args.seed)
        checkpoint_dir = args.checkpoint_dir.resolve()
        checkpoint_config = load_recover_config(checkpoint_dir)
        base_model_path = resolve_infer_value(
            cli_value=args.base_model_path,
            default_value=DEFAULT_BASE_MODEL_PATH,
            checkpoint_value=checkpoint_config.get("base_model_path"),
        )
        weight_dtype_name = resolve_infer_value(
            cli_value=args.weight_dtype,
            default_value=DEFAULT_WEIGHT_DTYPE,
            checkpoint_value=checkpoint_config.get("weight_dtype"),
        )
        codec_config = CodecConfig(**checkpoint_config["codec_config"])
        history_sizes = normalize_history_sizes(checkpoint_config["history_sizes"])
        latent_window_size = int(checkpoint_config["latent_window_size"])
        weight_dtype = parse_weight_dtype(str(weight_dtype_name))

        input_root, latent_paths = discover_input_latent_paths(args.input_path)
        output_dir = args.output_dir.resolve()
        low_dir = output_dir / "low_latents"
        recover_dir = output_dir / "recover_latents"
        metrics_dir = output_dir / "metrics"

        if distributed_context.is_main_process:
            low_dir.mkdir(parents=True, exist_ok=True)
            recover_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[infer] input_root={input_root} files={len(latent_paths)} "
                f"device={device} world_size={distributed_context.world_size}"
            )
        distributed_barrier(distributed_context)

        transformer, scheduler, prompt_embeds = load_transformer_bundle(
            base_model_path=str(base_model_path),
            device=device,
            weight_dtype=weight_dtype,
            checkpoint_dir=checkpoint_dir,
            gradient_checkpointing=False,
        )
        transformer.eval()

        for sample_idx, latent_path in enumerate(latent_paths, start=1):
            if distributed_context.is_main_process:
                print(f"[infer] ({sample_idx}/{len(latent_paths)}) input={latent_path}")

            sequence = prepare_sequence(latent_path, codec_config)
            recovered_full_latents = reconstruct_sequence(
                sequence=sequence,
                transformer=transformer,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                device=device,
                weight_dtype=weight_dtype,
                history_sizes=history_sizes,
                latent_window_size=latent_window_size,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                distributed_context=distributed_context,
            )

            if distributed_context.is_main_process:
                if recovered_full_latents is None:
                    raise RuntimeError("Main process did not receive reconstructed latent windows.")

                low_latent_path, recover_latent_path, metrics_path = resolve_output_paths(
                    input_path=latent_path,
                    input_root=input_root,
                    low_dir=low_dir,
                    recover_dir=recover_dir,
                    metrics_dir=metrics_dir,
                )
                low_payload = build_low_latent_payload(sequence)
                recover_payload = build_recover_latent_payload(
                    sequence=sequence,
                    recovered_full_latents=recovered_full_latents,
                    base_model_path=str(base_model_path),
                    checkpoint_dir=checkpoint_dir,
                )
                torch.save(low_payload, low_latent_path)
                torch.save(recover_payload, recover_latent_path)

                direct_low_metrics = compute_tensor_metrics(sequence.low_full_latents, sequence.clean_full_latents)
                restored_metrics = compute_tensor_metrics(recovered_full_latents, sequence.clean_full_latents)
                boundary_l1 = compute_boundary_transition_l1(
                    prediction=recovered_full_latents,
                    target=sequence.clean_full_latents,
                    latent_window_size=latent_window_size,
                )
                metrics_payload = {
                    "input_path": str(sequence.path),
                    "low_latent_path": str(low_latent_path),
                    "recover_latent_path": str(recover_latent_path),
                    "raw_file_bytes": sequence.metadata.raw_file_bytes,
                    "raw_bpp": sequence.metadata.raw_bpp,
                    "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
                    "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
                    "direct_low_metrics": direct_low_metrics,
                    "restored_metrics": restored_metrics,
                    "boundary_transition_l1": boundary_l1,
                    "codec_config": asdict(codec_config),
                    "checkpoint_dir": str(checkpoint_dir),
                    "num_inference_steps": args.num_inference_steps,
                }
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                with metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(metrics_payload, handle, indent=2)

                print(
                    f"[infer] saved low={low_latent_path} recover={recover_latent_path} "
                    f"low_bpp={sequence.low_codec_payload['low_bpp']:.6f} "
                    f"direct_low_l1={direct_low_metrics['l1']:.6f} restored_l1={restored_metrics['l1']:.6f}"
                )

            distributed_barrier(distributed_context)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        cleanup_distributed_context(distributed_context)


def build_codec_config(args: argparse.Namespace) -> CodecConfig:
    if args.temporal_factor < 1 or args.spatial_factor < 1:
        raise ValueError("temporal_factor and spatial_factor must both be >= 1.")
    return CodecConfig(
        temporal_factor=args.temporal_factor,
        spatial_factor=args.spatial_factor,
        quant_dtype=args.quant_dtype,
        keyframe_dtype=args.keyframe_dtype,
    )


def normalize_history_sizes(history_sizes: Sequence[int]) -> List[int]:
    normalized = sorted([int(value) for value in history_sizes], reverse=True)
    if sum(normalized) <= 0:
        raise ValueError("history_sizes must sum to a positive value.")
    return normalized


def discover_input_latent_paths(input_path: Path) -> Tuple[Path, List[Path]]:
    resolved = input_path.resolve()
    if resolved.is_file():
        if resolved.suffix != ".pt":
            raise ValueError(f"Expected a .pt latent file, got: {resolved}")
        return resolved.parent, [resolved]

    if not resolved.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Input path must be a latent file or directory: {resolved}")

    latent_paths = sorted(path for path in resolved.rglob("*.pt") if path.is_file())
    if not latent_paths:
        raise FileNotFoundError(f"No latent files found in: {resolved}")
    return resolved, latent_paths


def prepare_sequence(path: Path, codec_config: CodecConfig) -> PreparedSequence:
    payload = load_payload(path)
    validate_payload(payload, path)
    clean_full_latents = flatten_latent_chunks(payload["latent_chunks"]).float().contiguous()
    chunk_lengths = [int(chunk.shape[1]) for chunk in payload["latent_chunks"]]
    chunk_frame_ranges = build_chunk_frame_ranges(
        [int((chunk.shape[1] - 1) * 4 + 1) for chunk in payload["latent_chunks"]]
    )
    source_num_frames = infer_source_num_frames(payload["latent_chunks"], payload)
    source_width, source_height = resolve_source_resolution(payload, path)
    total_pixels = source_num_frames * source_width * source_height
    raw_file_bytes = path.stat().st_size
    raw_bpp = compute_bpp_from_total_pixels(raw_file_bytes, total_pixels)
    low_codec_payload = encode_low_latents(clean_full_latents, codec_config, chunk_lengths)
    low_codec_payload["low_codec_bytes"] = estimate_low_codec_bytes(low_codec_payload)
    low_codec_payload["low_bpp"] = compute_bpp_from_total_pixels(
        int(low_codec_payload["low_codec_bytes"]),
        total_pixels,
    )
    low_full_latents = decode_low_latents(low_codec_payload).float().contiguous()
    metadata = SequenceMetadata(
        source_fps=float(payload["source_fps"]),
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
        path=path.resolve(),
        metadata=metadata,
        clean_full_latents=clean_full_latents,
        low_codec_payload=low_codec_payload,
        low_full_latents=low_full_latents,
    )


def encode_low_latents(
    clean_full_latents: torch.Tensor,
    codec_config: CodecConfig,
    chunk_lengths: Sequence[int],
) -> Dict[str, object]:
    if clean_full_latents.ndim != 4:
        raise ValueError(f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}")

    keyframe_dtype = torch.float16 if codec_config.keyframe_dtype == "float16" else torch.float32
    keyframe = clean_full_latents[:, :1].to(keyframe_dtype).contiguous()
    remainder = clean_full_latents[:, 1:]

    if remainder.shape[1] == 0:
        return {
            "codec_config": asdict(codec_config),
            "chunk_lengths": [int(length) for length in chunk_lengths],
            "keyframe": keyframe.cpu(),
            "quantized_remainder": torch.empty(0, dtype=torch.int8),
            "scales": torch.empty(0, dtype=torch.float32),
            "reduced_shape": [int(clean_full_latents.shape[0]), 0, int(clean_full_latents.shape[2]), int(clean_full_latents.shape[3])],
            "original_remainder_shape": [int(value) for value in remainder.shape],
        }

    reduced_remainder = trilinear_resize(
        remainder.unsqueeze(0),
        (
            max(1, math.ceil(remainder.shape[1] / codec_config.temporal_factor)),
            max(1, math.ceil(remainder.shape[2] / codec_config.spatial_factor)),
            max(1, math.ceil(remainder.shape[3] / codec_config.spatial_factor)),
        ),
    ).squeeze(0)
    quantized_remainder, scales = symmetric_quantize_per_channel(reduced_remainder, codec_config.quant_dtype)
    return {
        "codec_config": asdict(codec_config),
        "chunk_lengths": [int(length) for length in chunk_lengths],
        "keyframe": keyframe.cpu(),
        "quantized_remainder": quantized_remainder.cpu(),
        "scales": scales.cpu(),
        "reduced_shape": [int(value) for value in reduced_remainder.shape],
        "original_remainder_shape": [int(value) for value in remainder.shape],
    }


def decode_low_latents(codec_payload: Dict[str, object]) -> torch.Tensor:
    keyframe = codec_payload["keyframe"].float()
    quantized_remainder = codec_payload["quantized_remainder"]
    original_remainder_shape = tuple(int(value) for value in codec_payload["original_remainder_shape"])
    if isinstance(quantized_remainder, torch.Tensor) and quantized_remainder.numel() == 0:
        return keyframe.float()

    reduced_remainder = symmetric_dequantize_per_channel(
        quantized_remainder=codec_payload["quantized_remainder"],
        scales=codec_payload["scales"],
    )
    restored_remainder = trilinear_resize(
        reduced_remainder.unsqueeze(0),
        (original_remainder_shape[1], original_remainder_shape[2], original_remainder_shape[3]),
    ).squeeze(0)
    return torch.cat([keyframe.float(), restored_remainder.float()], dim=1)


def symmetric_quantize_per_channel(latents: torch.Tensor, quant_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if quant_dtype != "int8":
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}. Only int8 is implemented.")
    scales = latents.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(EPS) / 127.0
    quantized = torch.round(latents / scales).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.squeeze(-1).squeeze(-1).squeeze(-1).float().contiguous()


def symmetric_dequantize_per_channel(quantized_remainder: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if quantized_remainder.ndim != 4:
        raise ValueError(
            f"Expected quantized remainder with shape [C, T, H, W], got {tuple(quantized_remainder.shape)}"
        )
    scale_view = scales.view(-1, 1, 1, 1).to(dtype=torch.float32)
    return quantized_remainder.float() * scale_view


def trilinear_resize(latents: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(latents.float(), size=size, mode="trilinear", align_corners=False)


def estimate_low_codec_bytes(codec_payload: Dict[str, object]) -> int:
    total_bytes = 0
    for key in ("keyframe", "quantized_remainder", "scales"):
        value = codec_payload[key]
        if isinstance(value, torch.Tensor):
            total_bytes += value.numel() * value.element_size()

    # Count only the minimal integer shape metadata needed to decode the payload.
    metadata_values: List[int] = [len(codec_payload["chunk_lengths"])]
    metadata_values.extend(int(value) for value in codec_payload["chunk_lengths"])
    metadata_values.extend(int(value) for value in codec_payload["reduced_shape"])
    metadata_values.extend(int(value) for value in codec_payload["original_remainder_shape"])
    total_bytes += len(metadata_values) * 4
    return int(total_bytes)


def extract_target_window(
    clean_full_latents: torch.Tensor,
    section_start: int,
    latent_window_size: int,
) -> Tuple[torch.Tensor, int]:
    target = clean_full_latents[:, section_start : section_start + latent_window_size]
    valid_target_frames = target.shape[1]
    if valid_target_frames == 0:
        raise ValueError(f"Section start {section_start} produced an empty target window.")
    if valid_target_frames < latent_window_size:
        padding = target[:, -1:].repeat(1, latent_window_size - valid_target_frames, 1, 1)
        target = torch.cat([target, padding], dim=1)
    return target.contiguous(), valid_target_frames


def extract_history_window(
    low_full_latents: torch.Tensor,
    section_start: int,
    history_window_size: int,
) -> torch.Tensor:
    low_remainder = low_full_latents[:, 1:]
    remainder_start = max(0, (section_start - 1) - history_window_size)
    remainder_end = max(0, section_start - 1)
    history = low_remainder[:, remainder_start:remainder_end]
    if history.shape[1] < history_window_size:
        zeros = torch.zeros(
            history.shape[0],
            history_window_size - history.shape[1],
            history.shape[2],
            history.shape[3],
            dtype=history.dtype,
        )
        history = torch.cat([zeros, history], dim=1)
    return history.contiguous()


def load_transformer_bundle(
    base_model_path: str,
    device: torch.device,
    weight_dtype: torch.dtype,
    checkpoint_dir: Optional[Path],
    gradient_checkpointing: bool,
    accelerator=None,
) -> Tuple[torch.nn.Module, object, torch.Tensor]:
    HeliosTransformer3DModel, HeliosScheduler, encode_prompt = import_helios_inference_stack()
    zero3_disabled = nullcontext()
    if accelerator is not None and accelerator.state.deepspeed_plugin is not None:
        # The prompt text encoder is only used to produce a shared empty-prompt embedding and
        # the transformer is later sharded by `accelerator.prepare(...)`, so disable ZeRO-3 init
        # during `from_pretrained` to keep Hugging Face model loading behavior intact.
        zero3_disabled = accelerator.state.deepspeed_plugin.zero3_init_context_manager(enable=False)

    with zero3_disabled:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
        ).to(device)
        with torch.inference_mode():
            prompt_embeds, _ = encode_prompt(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                prompt="",
                device=device,
                dtype=weight_dtype,
            )
        del tokenizer
        text_encoder.to("cpu")
        del text_encoder

        transformer = HeliosTransformer3DModel.from_pretrained(
            base_model_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
        )
    transformer.to(device)
    scheduler = HeliosScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    if checkpoint_dir is None:
        transformer.requires_grad_(True)
    else:
        state_dict_path = checkpoint_dir / "transformer_full.pt"
        if not state_dict_path.exists():
            raise FileNotFoundError(f"Missing transformer checkpoint: {state_dict_path}")
        state_dict = load_state_dict_file(state_dict_path)
        incompatible = transformer.load_state_dict(state_dict, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected state dict mismatch while loading {state_dict_path}: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        transformer.requires_grad_(False)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    return transformer, scheduler, prompt_embeds.to(device=device, dtype=weight_dtype)


def training_step(
    batch: Dict[str, torch.Tensor | int],
    transformer: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    history_sizes: Sequence[int],
    latent_window_size: int,
) -> torch.Tensor:
    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    history_latents = batch["history_latents"].to(device=device, dtype=weight_dtype)
    target_latents = batch["target_latents"].to(device=device, dtype=weight_dtype)
    x0_latents = batch["x0_latents"].to(device=device, dtype=weight_dtype)
    valid_target_frames = batch["valid_target_frames"].to(device=device)

    (
        model_input,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    ) = prepare_stage1_clean_input_from_latents(
        history_latents=history_latents,
        target_latents=target_latents,
        x0_latents=x0_latents,
        latent_window_size=latent_window_size,
        history_sizes=list(history_sizes),
        is_random_drop=False,
        random_drop_v2v_ratio=0.0,
        random_drop_t2v_ratio=0.0,
        is_keep_x0=True,
        dtype=weight_dtype,
        device=device,
    )

    noise = torch.randn(model_input.shape, device=device, dtype=weight_dtype)
    sigma = torch.rand(model_input.shape[0], device=device, dtype=weight_dtype) * 0.999
    sigma_view = sigma.view(-1, 1, 1, 1, 1)
    noisy_model_input = (1.0 - sigma_view) * model_input + sigma_view * noise
    target = noise - model_input
    timesteps = sigma * 1000.0

    base_transformer = unwrap_model(transformer)
    prompt_batch = prompt_embeds.repeat(model_input.shape[0], 1, 1)
    with base_transformer.cache_context("cond"):
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_batch,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )[0]

    mask = build_valid_mask(
        valid_target_frames=valid_target_frames,
        latent_window_size=latent_window_size,
        channels=model_pred.shape[1],
        height=model_pred.shape[3],
        width=model_pred.shape[4],
        device=device,
        dtype=model_pred.dtype,
    )
    return ((model_pred - target).float().pow(2) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def build_valid_mask(
    valid_target_frames: torch.Tensor,
    latent_window_size: int,
    channels: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size = valid_target_frames.shape[0]
    mask = torch.zeros(batch_size, 1, latent_window_size, 1, 1, device=device, dtype=dtype)
    for idx, valid in enumerate(valid_target_frames.tolist()):
        mask[idx, :, : int(valid)] = 1
    return mask * float(channels * height * width)


@torch.inference_mode()
def reconstruct_sequence(
    sequence: PreparedSequence,
    transformer: torch.nn.Module,
    scheduler: object,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    history_sizes: Sequence[int],
    latent_window_size: int,
    num_inference_steps: int,
    seed: int,
    distributed_context: DistributedContext,
) -> Optional[torch.Tensor]:
    clean_full = sequence.clean_full_latents
    low_full = sequence.low_full_latents
    if clean_full.shape[1] == 1:
        if distributed_context.is_main_process:
            return clean_full.clone().cpu().contiguous()
        return None

    prepare_stage1_clean_input_from_latents = import_stage1_prepare_fn()
    history_sizes = list(history_sizes)
    history_window_size = sum(history_sizes)
    section_starts = list(range(1, clean_full.shape[1], latent_window_size))
    assigned_section_starts = section_starts[distributed_context.rank :: distributed_context.world_size]
    local_section_results: List[Tuple[int, torch.Tensor]] = []
    dummy_target = torch.zeros(
        1,
        clean_full.shape[0],
        latent_window_size,
        clean_full.shape[2],
        clean_full.shape[3],
        device=device,
        dtype=weight_dtype,
    )
    x0_latents = clean_full[:, :1].unsqueeze(0).to(device=device, dtype=weight_dtype)

    for section_start in tqdm(
        assigned_section_starts,
        desc="Reconstructing local sections",
        disable=not distributed_context.is_main_process,
    ):
        history_latents = extract_history_window(
            low_full_latents=low_full,
            section_start=section_start,
            history_window_size=history_window_size,
        ).unsqueeze(0)
        valid_target_frames = min(latent_window_size, clean_full.shape[1] - section_start)

        (
            _,
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        ) = prepare_stage1_clean_input_from_latents(
            history_latents=history_latents.to(device=device, dtype=weight_dtype),
            target_latents=dummy_target,
            x0_latents=x0_latents,
            latent_window_size=latent_window_size,
            history_sizes=history_sizes,
            is_random_drop=False,
            random_drop_v2v_ratio=0.0,
            random_drop_t2v_ratio=0.0,
            is_keep_x0=True,
            dtype=weight_dtype,
            device=device,
        )

        section_latents = run_stage1_denoise(
            transformer=transformer,
            scheduler=scheduler,
            prompt_embeds=prompt_embeds,
            device=device,
            weight_dtype=weight_dtype,
            latent_shape=(1, clean_full.shape[0], latent_window_size, clean_full.shape[2], clean_full.shape[3]),
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            num_inference_steps=num_inference_steps,
            seed=seed + section_start,
        )
        local_section_results.append((section_start, section_latents[0, :, :valid_target_frames].contiguous()))

    gathered_sections = gather_reconstructed_sections(local_section_results, distributed_context)
    if gathered_sections is None:
        return None

    section_map: Dict[int, torch.Tensor] = {}
    for section_start, recovered_latents in gathered_sections:
        if section_start in section_map:
            raise RuntimeError(f"Duplicate reconstructed section received for section_start={section_start}.")
        section_map[section_start] = recovered_latents

    missing_sections = [section_start for section_start in section_starts if section_start not in section_map]
    if missing_sections:
        raise RuntimeError(f"Missing reconstructed sections on rank 0: {missing_sections}.")

    recovered_sections = [section_map[section_start] for section_start in section_starts]
    recovered_remainder = torch.cat(recovered_sections, dim=1)
    recovered_remainder = recovered_remainder[:, : clean_full.shape[1] - 1]
    return torch.cat([clean_full[:, :1].cpu(), recovered_remainder], dim=1).contiguous()


def gather_reconstructed_sections(
    local_section_results: List[Tuple[int, torch.Tensor]],
    distributed_context: DistributedContext,
) -> Optional[List[Tuple[int, torch.Tensor]]]:
    if not distributed_context.is_distributed:
        return local_section_results

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Distributed reconstruction expected an initialized process group.")

    gathered_section_results: Optional[List[List[Tuple[int, torch.Tensor]]]] = None
    if distributed_context.is_main_process:
        gathered_section_results = [list() for _ in range(distributed_context.world_size)]

    dist.gather_object(
        local_section_results,
        object_gather_list=gathered_section_results,
        dst=0,
    )
    if not distributed_context.is_main_process:
        return None
    assert gathered_section_results is not None

    merged_section_results: List[Tuple[int, torch.Tensor]] = []
    for rank_sections in gathered_section_results:
        merged_section_results.extend(rank_sections)
    return merged_section_results


@torch.inference_mode()
def run_stage1_denoise(
    transformer: torch.nn.Module,
    scheduler: object,
    prompt_embeds: torch.Tensor,
    device: torch.device,
    weight_dtype: torch.dtype,
    latent_shape: Tuple[int, int, int, int, int],
    indices_hidden_states: torch.Tensor,
    indices_latents_history_short: torch.Tensor,
    indices_latents_history_mid: torch.Tensor,
    indices_latents_history_long: torch.Tensor,
    latents_history_short: torch.Tensor,
    latents_history_mid: torch.Tensor,
    latents_history_long: torch.Tensor,
    num_inference_steps: int,
    seed: int,
) -> torch.Tensor:
    base_transformer = unwrap_model(transformer)
    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device=device, mu=1.0)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(latent_shape, device=device, dtype=torch.float32, generator=generator)

    for timestep in scheduler.timesteps:
        timestep_batch = timestep.expand(latents.shape[0])
        latent_model_input = latents.to(weight_dtype)
        with base_transformer.cache_context("cond"):
            noise_pred = transformer(
                hidden_states=latent_model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=indices_latents_history_long,
                latents_history_short=latents_history_short.to(weight_dtype),
                latents_history_mid=latents_history_mid.to(weight_dtype),
                latents_history_long=latents_history_long.to(weight_dtype),
                return_dict=False,
            )[0]
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    if hasattr(base_transformer, "clear_kv_cache"):
        base_transformer.clear_kv_cache()
    return latents.cpu()


def build_low_latent_payload(sequence: PreparedSequence) -> Dict[str, object]:
    return {
        "format_version": LOW_LATENT_FORMAT_V1,
        "input_path": str(sequence.path),
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
        "chunk_lengths": list(sequence.metadata.chunk_lengths),
        "codec_config": sequence.low_codec_payload["codec_config"],
        "keyframe": sequence.low_codec_payload["keyframe"],
        "quantized_remainder": sequence.low_codec_payload["quantized_remainder"],
        "scales": sequence.low_codec_payload["scales"],
        "reduced_shape": list(sequence.low_codec_payload["reduced_shape"]),
        "original_remainder_shape": list(sequence.low_codec_payload["original_remainder_shape"]),
        "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
        "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
    }


def build_recover_latent_payload(
    sequence: PreparedSequence,
    recovered_full_latents: torch.Tensor,
    base_model_path: str,
    checkpoint_dir: Path,
) -> Dict[str, object]:
    restored_chunks = split_full_latents(recovered_full_latents, sequence.metadata.chunk_lengths)
    return {
        "format_version": LATENT_FORMAT_V2,
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
        "chunk_frame_ranges": [list(frame_range) for frame_range in sequence.metadata.chunk_frame_ranges],
        "latent_chunks": restored_chunks,
        "model_path": str(base_model_path),
        "recovery_metadata": {
            "input_path": str(sequence.path),
            "checkpoint_dir": str(checkpoint_dir),
            "raw_bpp": sequence.metadata.raw_bpp,
            "low_codec_bytes": int(sequence.low_codec_payload["low_codec_bytes"]),
            "low_bpp": float(sequence.low_codec_payload["low_bpp"]),
        },
    }


def resolve_output_paths(
    input_path: Path,
    input_root: Path,
    low_dir: Path,
    recover_dir: Path,
    metrics_dir: Path,
) -> Tuple[Path, Path, Path]:
    relative_path = input_path.relative_to(input_root)
    low_latent_path = (low_dir / relative_path).with_suffix(".pt")
    recover_latent_path = (recover_dir / relative_path).with_suffix(".pt")
    metrics_path = (metrics_dir / relative_path).with_suffix(".json")
    low_latent_path.parent.mkdir(parents=True, exist_ok=True)
    recover_latent_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    return low_latent_path, recover_latent_path, metrics_path


def compute_tensor_metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    diff = (prediction.float() - target.float()).flatten()
    mse = float(diff.pow(2).mean().item())
    l1 = float(diff.abs().mean().item())
    psnr = float("inf") if mse <= EPS else float(-10.0 * math.log10(mse + EPS))
    return {
        "mse": mse,
        "l1": l1,
        "psnr": psnr,
    }


def compute_boundary_transition_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    latent_window_size: int,
) -> float:
    boundary_errors = []
    for section_start in range(1 + latent_window_size, prediction.shape[1], latent_window_size):
        pred_delta = prediction[:, section_start] - prediction[:, section_start - 1]
        target_delta = target[:, section_start] - target[:, section_start - 1]
        boundary_errors.append((pred_delta - target_delta).abs().mean().item())
    if not boundary_errors:
        return 0.0
    return float(sum(boundary_errors) / len(boundary_errors))


def save_training_artifacts(
    output_dir: Path,
    transformer: torch.nn.Module,
    codec_config: CodecConfig,
    history_sizes: Sequence[int],
    latent_window_size: int,
    args: argparse.Namespace,
    train_metrics: Dict[str, object],
    accelerator=None,
) -> None:
    if accelerator is not None:
        accelerator.wait_for_everyone()

    save_transformer_checkpoint(transformer, output_dir / "transformer_full.pt", accelerator=accelerator)

    recover_config = {
        "format_version": RECOVER_CONFIG_VERSION,
        "base_model_path": args.base_model_path,
        "weight_dtype": args.weight_dtype,
        "history_sizes": list(history_sizes),
        "latent_window_size": latent_window_size,
        "codec_config": asdict(codec_config),
    }
    if accelerator is None or accelerator.is_main_process:
        with (output_dir / "recover_config.json").open("w", encoding="utf-8") as handle:
            json.dump(recover_config, handle, indent=2)
        with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(train_metrics, handle, indent=2)

    if accelerator is not None:
        accelerator.wait_for_everyone()


def save_transformer_checkpoint(transformer: torch.nn.Module, output_path: Path, accelerator=None) -> None:
    if accelerator is not None:
        state_dict = accelerator.get_state_dict(transformer)
        if not accelerator.is_main_process:
            return
    else:
        state_dict = transformer.state_dict()
    cpu_state_dict = {}
    for key, value in state_dict.items():
        cpu_state_dict[key] = value.detach().cpu() if torch.is_tensor(value) else value
    torch.save(cpu_state_dict, output_path)


def load_recover_config(checkpoint_dir: Path) -> Dict[str, object]:
    config_path = checkpoint_dir / "recover_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing recover config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    required_keys = {"format_version", "history_sizes", "latent_window_size", "codec_config"}
    missing_keys = sorted(required_keys - config.keys())
    if missing_keys:
        raise KeyError(f"Recover config {config_path} is missing keys: {missing_keys}")
    if config["format_version"] != RECOVER_CONFIG_VERSION:
        raise ValueError(
            f"Unsupported recover config format in {config_path}: {config['format_version']}. "
            f"Expected {RECOVER_CONFIG_VERSION}."
        )
    return config


def resolve_infer_value(cli_value: object, default_value: object, checkpoint_value: object) -> object:
    if checkpoint_value is None:
        return cli_value
    if cli_value != default_value:
        return cli_value
    return checkpoint_value


def load_state_dict_file(path: Path) -> Dict[str, torch.Tensor]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parse_weight_dtype(weight_dtype: str) -> torch.dtype:
    if weight_dtype == "bf16":
        return torch.bfloat16
    if weight_dtype == "fp16":
        return torch.float16
    if weight_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported weight dtype: {weight_dtype}")


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


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
            raise RuntimeError("CUDA distributed training was requested but CUDA is not available.")
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
        dist.barrier()


def reduce_tensor_mean(tensor: torch.Tensor, distributed_context: DistributedContext) -> torch.Tensor:
    if not distributed_context.is_distributed:
        return tensor.detach().float()
    reduced = tensor.detach().float().clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= distributed_context.world_size
    return reduced


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = model
    while hasattr(unwrapped, "module"):
        next_model = getattr(unwrapped, "module")
        if next_model is unwrapped:
            break
        unwrapped = next_model
    return unwrapped


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
