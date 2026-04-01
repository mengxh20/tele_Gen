import argparse
import json
import math
import os
import random
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, UMT5EncoderModel

from reconstruct.decoder import (
    LATENT_FORMAT_V2,
    decode_latent_file_to_output_path,
    load_payload,
    resolve_device,
    validate_payload,
)
from reconstruct.encoder import compute_bpp_from_total_pixels

try:
    from peft import LoraConfig, set_peft_model_state_dict
    from peft.utils import get_peft_model_state_dict

    PEFT_IMPORT_ERROR = None
except ImportError as exc:
    LoraConfig = None
    set_peft_model_state_dict = None
    get_peft_model_state_dict = None
    PEFT_IMPORT_ERROR = exc


def import_helios_inference_stack():
    ensure_diffusers_parallel_shim()
    try:
        from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
        from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
        from helios.utils.utils_base import encode_prompt
    except ImportError as exc:
        raise ImportError(
            "Failed to import the Helios transformer stack. "
            "This environment's `diffusers` package is older than the version Helios expects. "
            "Please use the project's intended diffusers environment before running train/reconstruct."
        ) from exc

    return HeliosTransformer3DModel, HeliosScheduler, encode_prompt


def import_stage1_prepare_fn():
    from helios.utils.utils_helios_base import prepare_stage1_clean_input_from_latents

    return prepare_stage1_clean_input_from_latents


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


DEFAULT_BASE_MODEL_PATH = "/gemini/platform/public/luojx/team/mengxh/MODELS/BestWishYSH/Helios-Base"
DEFAULT_INPUT_DIR = "reconstruct/latents"
DEFAULT_OUTPUT_DIR = "reconstruct/gen2recon_runs"
DEFAULT_RECON_OUTPUT_DIR = "reconstruct/gen2recon_outputs"
DEFAULT_TEMPORAL_FACTOR = 2
DEFAULT_SPATIAL_FACTOR = 2
DEFAULT_QUANT_DTYPE = "int8"
DEFAULT_KEYFRAME_DTYPE = "float16"
DEFAULT_HISTORY_SIZES = [16, 2, 1]
DEFAULT_LATENT_WINDOW_SIZE = 9
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_WEIGHT_DTYPE = "bf16"
DEFAULT_LORA_TARGET_MODULES = ["attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0"]
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


@dataclass
class PreparedSequence:
    path: Path
    metadata: SequenceMetadata
    clean_full_latents: torch.Tensor
    low_codec_payload: Dict[str, torch.Tensor | int | str | Tuple[int, ...]]
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
        clean_full = sequence.clean_full_latents
        low_full = sequence.low_full_latents

        target_latents, valid_target_frames = extract_target_window(
            clean_full_latents=clean_full,
            section_start=section_start,
            latent_window_size=self.latent_window_size,
        )
        history_latents = extract_history_window(
            low_full_latents=low_full,
            section_start=section_start,
            history_window_size=self.history_window_size,
        )

        return {
            "history_latents": history_latents,
            "target_latents": target_latents,
            "x0_latents": clean_full[:, :1],
            "valid_target_frames": valid_target_frames,
            "seq_idx": seq_idx,
            "section_start": section_start,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keyframe-hybrid low-latent reconstruction POC.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Analyze raw latent bpp and low-latent codec stats.")
    add_common_codec_args(analyze_parser)
    analyze_parser.add_argument("--input_dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    analyze_parser.add_argument("--output_json", type=Path, default=None)
    analyze_parser.add_argument("--max_files", type=int, default=None)

    train_parser = subparsers.add_parser("train", help="Train LoRA adapters for low-latent to clean-latent recovery.")
    add_common_codec_args(train_parser)
    add_common_model_args(train_parser)
    train_parser.add_argument("--input_dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    train_parser.add_argument("--output_dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    train_parser.add_argument("--max_files", type=int, default=None)
    train_parser.add_argument("--batch_size", type=int, default=1)
    train_parser.add_argument("--epochs", type=int, default=500)
    train_parser.add_argument("--max_steps", type=int, default=None)
    train_parser.add_argument("--num_workers", type=int, default=8)
    train_parser.add_argument("--learning_rate", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=0.0)
    train_parser.add_argument("--max_grad_norm", type=float, default=1.0)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--log_every", type=int, default=10)
    train_parser.add_argument("--lora_rank", type=int, default=64)
    train_parser.add_argument("--lora_alpha", type=int, default=64)
    train_parser.add_argument("--lora_dropout", type=float, default=0.0)
    train_parser.add_argument("--weight_dtype", type=str, default=DEFAULT_WEIGHT_DTYPE, choices=["bf16", "fp16", "fp32"])
    train_parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    recon_parser = subparsers.add_parser("reconstruct", help="Reconstruct latent files from a directory of low-latent conditions.")
    add_common_codec_args(recon_parser)
    add_common_model_args(recon_parser)
    recon_parser.add_argument("--input_path", type=Path, required=True)
    recon_parser.add_argument("--lora_path", type=Path, required=True)
    recon_parser.add_argument("--output_latent_path", type=Path, required=True)
    recon_parser.add_argument("--output_video_path", type=Path, default=None)
    recon_parser.add_argument("--metrics_json", type=Path, default=None)
    recon_parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    recon_parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    recon_parser.add_argument("--seed", type=int, default=42)
    recon_parser.add_argument("--weight_dtype", type=str, default=DEFAULT_WEIGHT_DTYPE, choices=["bf16", "fp16", "fp32"])

    return parser.parse_args()


def add_common_codec_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--temporal_factor", type=int, default=DEFAULT_TEMPORAL_FACTOR)
    parser.add_argument("--spatial_factor", type=int, default=DEFAULT_SPATIAL_FACTOR)
    parser.add_argument("--quant_dtype", type=str, default=DEFAULT_QUANT_DTYPE, choices=["int8"])
    parser.add_argument("--keyframe_dtype", type=str, default=DEFAULT_KEYFRAME_DTYPE, choices=["float16", "float32"])
    parser.add_argument("--history_sizes", type=int, nargs="+", default=DEFAULT_HISTORY_SIZES)
    parser.add_argument("--latent_window_size", type=int, default=DEFAULT_LATENT_WINDOW_SIZE)


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base_model_path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--device", type=str, default="cuda")


def main() -> None:
    args = parse_args()
    if args.command == "analyze":
        command_analyze(args)
    elif args.command == "train":
        command_train(args)
    elif args.command == "reconstruct":
        command_reconstruct(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


def command_analyze(args: argparse.Namespace) -> None:
    codec_config = build_codec_config(args)
    latent_paths = discover_latent_paths(args.input_dir, max_files=args.max_files)
    prepared_sequences = [prepare_sequence(path, codec_config) for path in latent_paths]

    per_file_results = []
    for sequence in prepared_sequences:
        result = analyze_sequence(sequence)
        per_file_results.append(result)
        print(
            f"[analyze] path={sequence.path} raw_bpp={result['raw_bpp']:.6f} "
            f"low_bpp={result['low_bpp']:.6f} direct_low_l1={result['direct_low_l1']:.6f} "
            f"direct_low_mse={result['direct_low_mse']:.6f}"
        )

    summary = summarize_analysis(per_file_results)
    print(
        f"[analyze] summary files={summary['num_files']} avg_raw_bpp={summary['avg_raw_bpp']:.6f} "
        f"avg_low_bpp={summary['avg_low_bpp']:.6f} avg_direct_low_l1={summary['avg_direct_low_l1']:.6f}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "codec_config": asdict(codec_config),
            "files": per_file_results,
            "summary": summary,
        }
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def command_train(args: argparse.Namespace) -> None:
    ensure_peft_available()
    distributed_context, device = init_distributed_context(args.device)
    try:
        set_seed(args.seed + distributed_context.rank)
        weight_dtype = parse_weight_dtype(args.weight_dtype)
        codec_config = build_codec_config(args)
        history_sizes = normalize_history_sizes(args.history_sizes)

        latent_paths = discover_latent_paths(args.input_dir, max_files=args.max_files)
        prepared_sequences = [prepare_sequence(path, codec_config) for path in latent_paths]
        dataset = LatentWindowDataset(
            sequences=prepared_sequences,
            history_sizes=history_sizes,
            latent_window_size=args.latent_window_size,
        )
        if len(dataset) == 0:
            raise RuntimeError("No training windows were built from the provided latent files.")

        sampler: DistributedSampler | None = None
        if distributed_context.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=distributed_context.world_size,
                rank=distributed_context.rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=device.type == "cuda",
        )

        transformer, _scheduler, prompt_embeds = load_transformer_bundle(
            base_model_path=args.base_model_path,
            device=device,
            weight_dtype=weight_dtype,
            lora_path=None,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            gradient_checkpointing=args.gradient_checkpointing,
            enable_xformers_memory_efficient_attention=args.enable_xformers_memory_efficient_attention,
        )
        transformer.train()
        base_transformer = transformer
        if distributed_context.is_distributed:
            ddp_kwargs = {"broadcast_buffers": False}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [device.index]
                ddp_kwargs["output_device"] = device.index
            transformer = DistributedDataParallel(transformer, **ddp_kwargs)

        optimizer = torch.optim.AdamW(
            [param for param in transformer.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        output_dir = args.output_dir.resolve()
        if distributed_context.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)

        train_metrics = {
            "losses": [],
            "steps": 0,
            "trainable_params": count_trainable_parameters(base_transformer),
            "world_size": distributed_context.world_size,
            "per_device_batch_size": args.batch_size,
            "global_batch_size": args.batch_size * distributed_context.world_size,
            "gradient_checkpointing": args.gradient_checkpointing,
            "enable_xformers_memory_efficient_attention": args.enable_xformers_memory_efficient_attention,
        }
        if distributed_context.is_main_process:
            print(
                f"[train] world_size={distributed_context.world_size} "
                f"per_device_batch_size={args.batch_size} "
                f"global_batch_size={train_metrics['global_batch_size']} "
                f"device={device}"
            )

        progress = None
        if args.max_steps is not None and distributed_context.is_main_process:
            progress = tqdm(total=args.max_steps, desc="Training")

        global_step = 0
        stop_training = False
        for epoch in range(args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch + 1}/{args.epochs}",
                disable=not distributed_context.is_main_process,
            )
            for batch in epoch_iterator:
                optimizer.zero_grad(set_to_none=True)
                loss = training_step(
                    batch=batch,
                    transformer=transformer,
                    prompt_embeds=prompt_embeds,
                    device=device,
                    weight_dtype=weight_dtype,
                    history_sizes=history_sizes,
                    latent_window_size=args.latent_window_size,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()

                global_step += 1
                reduced_loss = reduce_tensor_mean(loss.detach(), distributed_context)
                if distributed_context.is_main_process:
                    train_metrics["steps"] = global_step
                    train_metrics["losses"].append(float(reduced_loss.cpu().item()))
                    if global_step % args.log_every == 0:
                        epoch_iterator.set_postfix(loss=f"{train_metrics['losses'][-1]:.6f}")
                    if progress is not None:
                        progress.update(1)

                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_training = True
                    break

            distributed_barrier(distributed_context)
            if distributed_context.is_main_process:
                save_training_artifacts(
                    output_dir=output_dir,
                    transformer=unwrap_model(transformer),
                    codec_config=codec_config,
                    args=args,
                    train_metrics=train_metrics,
                )
            distributed_barrier(distributed_context)
            if stop_training:
                break

        if progress is not None:
            progress.close()
        if distributed_context.is_main_process and train_metrics["losses"]:
            print(
                f"[train] completed steps={train_metrics['steps']} "
                f"last_loss={train_metrics['losses'][-1]:.6f} "
                f"trainable_params={train_metrics['trainable_params']}"
            )
    finally:
        cleanup_distributed_context(distributed_context)


def command_reconstruct(args: argparse.Namespace) -> None:
    distributed_context, device = init_distributed_context(args.device)
    try:
        set_seed(args.seed)
        weight_dtype = parse_weight_dtype(args.weight_dtype)
        codec_config = build_codec_config(args)
        history_sizes = normalize_history_sizes(args.history_sizes)
        input_dir = resolve_directory_argument(args.input_path, "--input_path", must_exist=True)
        output_latent_dir = resolve_directory_argument(args.output_latent_path, "--output_latent_path", must_exist=False)
        output_video_dir = (
            resolve_directory_argument(args.output_video_path, "--output_video_path", must_exist=False)
            if args.output_video_path is not None
            else None
        )
        metrics_dir = (
            output_latent_dir
            if args.metrics_json is None
            else resolve_directory_argument(args.metrics_json, "--metrics_json", must_exist=False)
        )
        latent_paths = discover_latent_paths(input_dir)
        validate_flat_reconstruct_outputs(latent_paths)
        ensure_peft_available()

        if distributed_context.is_main_process:
            print(
                f"[reconstruct] world_size={distributed_context.world_size} "
                f"device={device} input_dir={input_dir} num_files={len(latent_paths)}"
            )
            output_latent_dir.mkdir(parents=True, exist_ok=True)
            if output_video_dir is not None:
                output_video_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            print(f"[reconstruct] output_latent_dir={output_latent_dir}")
            print(f"[reconstruct] metrics_dir={metrics_dir}")
            if output_video_dir is not None:
                print(f"[reconstruct] output_video_dir={output_video_dir}")
        distributed_barrier(distributed_context)

        transformer, scheduler, prompt_embeds = load_transformer_bundle(
            base_model_path=args.base_model_path,
            device=device,
            weight_dtype=weight_dtype,
            lora_path=args.lora_path.resolve(),
            lora_rank=None,
            lora_alpha=None,
            lora_dropout=None,
            gradient_checkpointing=False,
            enable_xformers_memory_efficient_attention=False,
        )
        transformer.eval()
        decoder_model_bundle_cache = {}

        for sample_idx, latent_path in enumerate(latent_paths, start=1):
            if distributed_context.is_main_process:
                print(f"[reconstruct] ({sample_idx}/{len(latent_paths)}) input={latent_path}")

            sequence = prepare_sequence(latent_path, codec_config)
            restored_full_latents = reconstruct_sequence(
                sequence=sequence,
                transformer=transformer,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                device=device,
                weight_dtype=weight_dtype,
                history_sizes=history_sizes,
                latent_window_size=args.latent_window_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                distributed_context=distributed_context,
            )

            if distributed_context.is_main_process:
                if restored_full_latents is None:
                    raise RuntimeError("Main process did not receive reconstructed latent windows.")

                output_latent_path, metrics_path, output_video_path = resolve_reconstruct_sample_paths(
                    input_path=sequence.path,
                    output_latent_dir=output_latent_dir,
                    metrics_dir=metrics_dir,
                    output_video_dir=output_video_dir,
                )
                baseline_metrics = compute_tensor_metrics(sequence.low_full_latents, sequence.clean_full_latents)
                restored_metrics = compute_tensor_metrics(restored_full_latents, sequence.clean_full_latents)
                boundary_l1 = compute_boundary_transition_l1(
                    prediction=restored_full_latents,
                    target=sequence.clean_full_latents,
                    latent_window_size=args.latent_window_size,
                )
                analysis = analyze_sequence(sequence)

                output_payload = build_output_payload(sequence, restored_full_latents)
                torch.save(output_payload, output_latent_path)
                print(f"[reconstruct] saved latent payload to {output_latent_path}")

                if output_video_path is not None:
                    decode_latent_file_to_output_path(
                        latent_path=output_latent_path,
                        output_path=output_video_path,
                        cli_model_path=args.base_model_path,
                        model_bundle_cache=decoder_model_bundle_cache,
                        device=device,
                    )

                metrics_payload = {
                    "input_path": str(sequence.path),
                    "output_latent_path": str(output_latent_path),
                    "output_video_path": str(output_video_path) if output_video_path is not None else None,
                    "codec_config": asdict(codec_config),
                    "analysis": analysis,
                    "direct_low_metrics": baseline_metrics,
                    "restored_metrics": restored_metrics,
                    "boundary_transition_l1": boundary_l1,
                    "num_inference_steps": args.num_inference_steps,
                    "world_size": distributed_context.world_size,
                }
                with metrics_path.open("w", encoding="utf-8") as handle:
                    json.dump(metrics_payload, handle, indent=2)
                print(f"[reconstruct] saved metrics to {metrics_path}")

                print(
                    f"[reconstruct] raw_bpp={analysis['raw_bpp']:.6f} low_bpp={analysis['low_bpp']:.6f} "
                    f"direct_low_l1={baseline_metrics['l1']:.6f} restored_l1={restored_metrics['l1']:.6f} "
                    f"boundary_transition_l1={boundary_l1:.6f}"
                )

            restored_full_latents = None
            sequence = None
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


def resolve_directory_argument(path: Path, argument_name: str, must_exist: bool) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"{argument_name} must be a directory, got file path: {resolved}")
        return resolved
    if must_exist:
        raise FileNotFoundError(f"{argument_name} does not exist: {resolved}")
    if resolved.suffix:
        raise ValueError(f"{argument_name} must be a directory path, got file-like path: {path}")
    return resolved


def build_reconstruct_output_stem(input_path: Path) -> str:
    return f"{input_path.stem}_reconstructed"


def validate_flat_reconstruct_outputs(input_paths: Sequence[Path]) -> None:
    stem_to_paths: Dict[str, List[Path]] = {}
    for input_path in input_paths:
        output_stem = build_reconstruct_output_stem(input_path)
        stem_to_paths.setdefault(output_stem, []).append(input_path)

    duplicate_outputs = {stem: paths for stem, paths in stem_to_paths.items() if len(paths) > 1}
    if duplicate_outputs:
        conflict_lines = []
        for stem, paths in sorted(duplicate_outputs.items()):
            joined_paths = ", ".join(str(path) for path in paths)
            conflict_lines.append(f"{stem}: {joined_paths}")
        raise ValueError(
            "Flat reconstruct outputs would overwrite each other because multiple inputs share the same stem. "
            f"Conflicts: {'; '.join(conflict_lines)}"
        )


def resolve_reconstruct_sample_paths(
    input_path: Path,
    output_latent_dir: Path,
    metrics_dir: Path,
    output_video_dir: Optional[Path],
) -> Tuple[Path, Path, Optional[Path]]:
    output_stem = build_reconstruct_output_stem(input_path)
    output_latent_path = output_latent_dir / f"{output_stem}.pt"
    metrics_path = metrics_dir / f"{output_stem}.metrics.json"
    output_video_path = output_video_dir / f"{output_stem}.mp4" if output_video_dir is not None else None
    return output_latent_path, metrics_path, output_video_path


def ensure_peft_available() -> None:
    if PEFT_IMPORT_ERROR is not None:
        raise ImportError(
            "The `peft` package is required for the train/reconstruct commands. "
            "Install it in this environment before using LoRA features."
        ) from PEFT_IMPORT_ERROR


def normalize_history_sizes(history_sizes: Sequence[int]) -> List[int]:
    history_sizes = sorted(list(history_sizes), reverse=True)
    if sum(history_sizes) <= 0:
        raise ValueError("history_sizes must sum to a positive value.")
    return history_sizes


def discover_latent_paths(input_dir: Path, max_files: Optional[int] = None) -> List[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    latent_paths = sorted(path for path in input_dir.rglob("*.pt") if path.is_file())
    if max_files is not None:
        latent_paths = latent_paths[:max_files]
    if not latent_paths:
        raise FileNotFoundError(f"No latent files found in: {input_dir}")
    return latent_paths


def prepare_sequence(path: Path, codec_config: CodecConfig) -> PreparedSequence:
    payload = load_payload(path)
    validate_payload(payload, path)
    clean_full_latents = flatten_latent_chunks(payload["latent_chunks"]).float().contiguous()
    chunk_lengths = [chunk.shape[1] for chunk in payload["latent_chunks"]]
    source_num_frames = infer_source_num_frames(payload["latent_chunks"], payload)
    source_width, source_height = parse_source_resolution(path)
    total_pixels = source_num_frames * source_width * source_height
    raw_file_bytes = path.stat().st_size
    raw_bpp = compute_bpp_from_total_pixels(raw_file_bytes, total_pixels)
    low_codec_payload = encode_low_latents(clean_full_latents, codec_config)
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
    )
    return PreparedSequence(
        path=path,
        metadata=metadata,
        clean_full_latents=clean_full_latents,
        low_codec_payload=low_codec_payload,
        low_full_latents=low_full_latents,
    )


def flatten_latent_chunks(latent_chunks: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([chunk.float().contiguous() for chunk in latent_chunks], dim=1)


def infer_source_num_frames(latent_chunks: Sequence[torch.Tensor], payload: Dict) -> int:
    source_num_frames = payload.get("source_num_frames")
    if source_num_frames is not None:
        return int(source_num_frames)
    return sum((chunk.shape[1] - 1) * 4 + 1 for chunk in latent_chunks)


def parse_source_resolution(path: Path) -> Tuple[int, int]:
    stem_parts = path.stem.split("_")
    if len(stem_parts) < 2 or "x" not in stem_parts[-2]:
        raise ValueError(
            f"Unable to infer source resolution from {path.name}. Expected suffix like *_1280x720_30.pt."
        )
    width_str, height_str = stem_parts[-2].split("x", maxsplit=1)
    return int(width_str), int(height_str)


def encode_low_latents(clean_full_latents: torch.Tensor, codec_config: CodecConfig) -> Dict[str, torch.Tensor | int | str | Tuple[int, ...]]:
    if clean_full_latents.ndim != 4:
        raise ValueError(f"Expected clean_full_latents with shape [C, T, H, W], got {tuple(clean_full_latents.shape)}")

    keyframe = clean_full_latents[:, :1]
    keyframe_dtype = torch.float16 if codec_config.keyframe_dtype == "float16" else torch.float32
    keyframe = keyframe.to(keyframe_dtype).contiguous()

    remainder = clean_full_latents[:, 1:]
    if remainder.shape[1] == 0:
        return {
            "codec_config": asdict(codec_config),
            "keyframe": keyframe.cpu(),
            "quantized_remainder": torch.empty(0, dtype=torch.int8),
            "scales": torch.empty(0, dtype=torch.float32),
            "reduced_shape": (clean_full_latents.shape[0], 0, clean_full_latents.shape[2], clean_full_latents.shape[3]),
            "original_remainder_shape": tuple(remainder.shape),
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
        "keyframe": keyframe.cpu(),
        "quantized_remainder": quantized_remainder.cpu(),
        "scales": scales.cpu(),
        "reduced_shape": tuple(reduced_remainder.shape),
        "original_remainder_shape": tuple(remainder.shape),
    }


def decode_low_latents(codec_payload: Dict[str, torch.Tensor | int | str | Tuple[int, ...]]) -> torch.Tensor:
    keyframe = codec_payload["keyframe"].float()
    quantized_remainder = codec_payload["quantized_remainder"]
    original_remainder_shape = tuple(codec_payload["original_remainder_shape"])
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
        raise ValueError(f"Unsupported quant_dtype: {quant_dtype}. Only int8 is implemented in this POC.")
    scales = latents.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(EPS) / 127.0
    quantized = torch.round(latents / scales).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.squeeze(-1).squeeze(-1).squeeze(-1).float().contiguous()


def symmetric_dequantize_per_channel(quantized_remainder: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if quantized_remainder.ndim != 4:
        raise ValueError(f"Expected quantized remainder with shape [C, T, H, W], got {tuple(quantized_remainder.shape)}")
    scale_view = scales.view(-1, 1, 1, 1).to(dtype=torch.float32)
    return quantized_remainder.float() * scale_view


def trilinear_resize(latents: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(latents.float(), size=size, mode="trilinear", align_corners=False)


def estimate_codec_bytes(codec_payload: Dict[str, torch.Tensor | int | str | Tuple[int, ...]]) -> int:
    keyframe = codec_payload["keyframe"]
    quantized_remainder = codec_payload["quantized_remainder"]
    scales = codec_payload["scales"]

    total_bytes = 0
    if isinstance(keyframe, torch.Tensor):
        total_bytes += keyframe.numel() * keyframe.element_size()
    if isinstance(quantized_remainder, torch.Tensor):
        total_bytes += quantized_remainder.numel() * quantized_remainder.element_size()
    if isinstance(scales, torch.Tensor):
        total_bytes += scales.numel() * scales.element_size()
    total_bytes += 4 * 8
    return int(total_bytes)


def analyze_sequence(sequence: PreparedSequence) -> Dict[str, float | int | str]:
    codec_bytes = estimate_codec_bytes(sequence.low_codec_payload)
    low_bpp = compute_bpp_from_total_pixels(codec_bytes, sequence.metadata.total_pixels)
    low_metrics = compute_tensor_metrics(sequence.low_full_latents, sequence.clean_full_latents)
    return {
        "path": str(sequence.path),
        "raw_file_bytes": sequence.metadata.raw_file_bytes,
        "raw_bpp": sequence.metadata.raw_bpp,
        "low_codec_bytes": codec_bytes,
        "low_bpp": low_bpp,
        "direct_low_l1": low_metrics["l1"],
        "direct_low_mse": low_metrics["mse"],
        "direct_low_psnr": low_metrics["psnr"],
        "source_num_frames": sequence.metadata.source_num_frames,
        "source_width": sequence.metadata.source_width,
        "source_height": sequence.metadata.source_height,
    }


def summarize_analysis(results: Sequence[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    if not results:
        return {
            "num_files": 0,
            "avg_raw_bpp": 0.0,
            "avg_low_bpp": 0.0,
            "avg_direct_low_l1": 0.0,
            "avg_direct_low_mse": 0.0,
        }
    return {
        "num_files": len(results),
        "avg_raw_bpp": float(sum(float(item["raw_bpp"]) for item in results) / len(results)),
        "avg_low_bpp": float(sum(float(item["low_bpp"]) for item in results) / len(results)),
        "avg_direct_low_l1": float(sum(float(item["direct_low_l1"]) for item in results) / len(results)),
        "avg_direct_low_mse": float(sum(float(item["direct_low_mse"]) for item in results) / len(results)),
    }


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
    lora_path: Optional[Path],
    lora_rank: Optional[int],
    lora_alpha: Optional[int],
    lora_dropout: Optional[float],
    gradient_checkpointing: bool = False,
    enable_xformers_memory_efficient_attention: bool = False,
) -> Tuple[torch.nn.Module, object, torch.Tensor]:
    HeliosTransformer3DModel, HeliosScheduler, encode_prompt = import_helios_inference_stack()
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
    transformer.requires_grad_(False)
    transformer.to(device)
    scheduler = HeliosScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    if lora_path is None:
        if lora_rank is None or lora_alpha is None or lora_dropout is None:
            raise ValueError("LoRA hyper-parameters must be provided when initializing a new adapter.")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights="gaussian",
            target_modules=DEFAULT_LORA_TARGET_MODULES,
        )
        transformer.add_adapter(lora_config)
        for param in transformer.parameters():
            if param.requires_grad:
                param.data = param.data.float()
    else:
        load_lora_checkpoint(transformer, lora_path)
        transformer.requires_grad_(False)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if enable_xformers_memory_efficient_attention:
        try:
            transformer.enable_xformers_memory_efficient_attention()
        except Exception as exc:
            raise RuntimeError(
                "xFormers memory efficient attention was requested, but it could not be enabled in this environment."
            ) from exc

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
    loss = ((model_pred - target).float().pow(2) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return loss


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
    guidance_scale: float,
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

    if distributed_context.is_main_process:
        print(
            f"[reconstruct] total_sections={len(section_starts)} "
            f"local_sections={len(assigned_section_starts)}"
        )

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
            guidance_scale=guidance_scale,
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
        raise RuntimeError(
            f"Missing reconstructed sections on rank 0: {missing_sections}. "
            "Distributed reconstruction did not produce a complete remainder."
        )

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
    guidance_scale: float,
    seed: int,
) -> torch.Tensor:
    if guidance_scale != 1.0:
        raise ValueError("This POC fixes guidance_scale=1.0 and does not implement classifier-free guidance.")

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


def build_output_payload(sequence: PreparedSequence, restored_full_latents: torch.Tensor) -> Dict:
    restored_chunks = split_full_latents(restored_full_latents, sequence.metadata.chunk_lengths)
    return {
        "format_version": LATENT_FORMAT_V2,
        "source_fps": sequence.metadata.source_fps,
        "source_num_frames": sequence.metadata.source_num_frames,
        "latent_chunks": restored_chunks,
        "gen2recon_metadata": {
            "input_path": str(sequence.path),
            "raw_bpp": sequence.metadata.raw_bpp,
            "low_codec_bytes": estimate_codec_bytes(sequence.low_codec_payload),
        },
    }


def split_full_latents(full_latents: torch.Tensor, chunk_lengths: Sequence[int]) -> List[torch.Tensor]:
    restored_chunks: List[torch.Tensor] = []
    start = 0
    for length in chunk_lengths:
        end = start + length
        restored_chunks.append(full_latents[:, start:end].contiguous())
        start = end
    return restored_chunks


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
    args: argparse.Namespace,
    train_metrics: Dict[str, float | int | List[float]],
) -> None:
    lora_state_dict = get_peft_model_state_dict(transformer)
    torch.save(lora_state_dict, output_dir / "transformer_lora.pt")

    lora_config = {
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": DEFAULT_LORA_TARGET_MODULES,
    }
    with (output_dir / "lora_config.json").open("w", encoding="utf-8") as handle:
        json.dump(lora_config, handle, indent=2)
    with (output_dir / "codec_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(codec_config), handle, indent=2)
    with (output_dir / "train_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(train_metrics, handle, indent=2)


def load_lora_checkpoint(transformer: torch.nn.Module, lora_path: Path) -> None:
    lora_config_path = lora_path / "lora_config.json"
    if not lora_config_path.exists():
        raise FileNotFoundError(f"Missing LoRA config: {lora_config_path}")
    with lora_config_path.open("r", encoding="utf-8") as handle:
        lora_config_dict = json.load(handle)
    lora_config = LoraConfig(
        r=int(lora_config_dict["r"]),
        lora_alpha=int(lora_config_dict["lora_alpha"]),
        lora_dropout=float(lora_config_dict["lora_dropout"]),
        init_lora_weights="gaussian",
        target_modules=list(lora_config_dict["target_modules"]),
    )
    transformer.add_adapter(lora_config)

    state_dict_path = lora_path / "transformer_lora.pt"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Missing LoRA state dict: {state_dict_path}")
    state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    incompatible_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name="default")
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None) if incompatible_keys is not None else None
    if unexpected_keys:
        raise RuntimeError(f"Unexpected LoRA keys while loading {state_dict_path}: {unexpected_keys}")


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
        raise RuntimeError("Distributed mode requires LOCAL_RANK to be set. Please launch training with torchrun.")

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

    return DistributedContext(
        is_distributed=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    ), device


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
    # Access Helios-specific helpers like `cache_context()` on the wrapped module.
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
