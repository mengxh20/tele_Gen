import json
import os
import zlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import zstandard as zstd
except ImportError:
    zstd = None


MAGIC = b"HLENTRPY"
VERSION = 2
FORMAT_NAME = "helios_entropy_latent_v2"
FIELD_STRATEGY = "per_field_compressed"

DTYPE_NAME_TO_NUMPY = {
    "int8": np.int8,
    "uint8": np.uint8,
    "float16": np.float16,
    "float32": np.float32,
    "int32": np.int32,
    "int64": np.int64,
}
DTYPE_NAME_TO_TORCH = {
    "int8": torch.int8,
    "uint8": torch.uint8,
    "float16": torch.float16,
    "float32": torch.float32,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _compress_bytes(raw: bytes) -> Tuple[str, bytes]:
    if zstd is not None:
        compressor = zstd.ZstdCompressor(level=9)
        return "zstd", compressor.compress(raw)
    return "zlib", zlib.compress(raw, level=9)


def _decompress_bytes(codec: str, value: bytes) -> bytes:
    if codec == "zstd":
        if zstd is None:
            raise RuntimeError("zstandard is required to decode zstd-compressed entropy payloads.")
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(value)
    if codec == "zlib":
        return zlib.decompress(value)
    raise ValueError(f"Unsupported external entropy codec={codec}.")


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype).replace("torch.", "")
    if name not in DTYPE_NAME_TO_NUMPY:
        raise ValueError(f"Unsupported tensor dtype for external entropy payload: {dtype}.")
    return name


def _tensor_to_bytes(tensor: torch.Tensor) -> Tuple[bytes, Dict[str, object]]:
    tensor_cpu = tensor.detach().cpu().contiguous()
    dtype_name = _dtype_name(tensor_cpu.dtype)
    array = tensor_cpu.numpy()
    return (
        array.tobytes(order="C"),
        {
            "dtype": dtype_name,
            "shape": [int(value) for value in tensor_cpu.shape],
            "numel": int(tensor_cpu.numel()),
            "raw_num_bytes": int(tensor_cpu.numel() * tensor_cpu.element_size()),
        },
    )


def _bytes_to_tensor(raw: bytes, descriptor: Dict[str, object]) -> torch.Tensor:
    dtype_name = str(descriptor["dtype"])
    if dtype_name not in DTYPE_NAME_TO_NUMPY or dtype_name not in DTYPE_NAME_TO_TORCH:
        raise ValueError(f"Unsupported tensor dtype in external entropy payload: {dtype_name}.")
    shape = tuple(int(value) for value in descriptor["shape"])
    array = np.frombuffer(raw, dtype=DTYPE_NAME_TO_NUMPY[dtype_name]).copy()
    if shape:
        array = array.reshape(shape)
    return torch.from_numpy(array).to(dtype=DTYPE_NAME_TO_TORCH[dtype_name]).contiguous()


def _build_source_payload(low_payload: Dict[str, object], relative_path: str | None) -> Dict[str, object]:
    return {
        "format_version": str(low_payload["format_version"]),
        "input_path": str(low_payload["input_path"]),
        "relative_path": relative_path,
        "source_fps": float(low_payload["source_fps"]),
        "source_num_frames": int(low_payload["source_num_frames"]),
        "source_width": int(low_payload["source_width"]),
        "source_height": int(low_payload["source_height"]),
        "chunk_lengths": [int(value) for value in low_payload["chunk_lengths"]],
        "codec_config_v2": low_payload["codec_config_v2"],
        "codec_config": low_payload["codec_config"],
        "tail_codec_type": str(low_payload["tail_codec_type"]),
        "section_ranges": [[int(value) for value in section_range] for section_range in low_payload["section_ranges"]],
        "low_codec_bytes": int(low_payload["low_codec_bytes"]),
        "low_bpp": float(low_payload["low_bpp"]),
    }


def _build_block(block_name: str, raw: bytes) -> Tuple[Dict[str, object], bytes]:
    codec_name, compressed = _compress_bytes(raw)
    descriptor = {
        "block": block_name,
        "codec": codec_name,
        "compressed_num_bytes": len(compressed),
        "raw_num_bytes": len(raw),
    }
    return descriptor, compressed


def write_external_entropy_payload(
    low_payload: Dict[str, object],
    output_path: Path,
    *,
    relative_path: str | None = None,
) -> Dict[str, object]:
    block_order: List[str] = []
    block_lengths: Dict[str, int] = {}
    block_payloads: List[bytes] = []

    def add_bytes_block(block_name: str, raw: bytes) -> Dict[str, object]:
        descriptor, compressed = _build_block(block_name, raw)
        block_order.append(block_name)
        block_lengths[block_name] = len(compressed)
        block_payloads.append(compressed)
        return descriptor

    def add_tensor_block(block_name: str, tensor: torch.Tensor) -> Dict[str, object]:
        raw, tensor_descriptor = _tensor_to_bytes(tensor)
        block_descriptor = add_bytes_block(block_name, raw)
        return {
            **tensor_descriptor,
            **block_descriptor,
        }

    def add_quantized_tensor_payload(block_name: str, payload: Dict[str, object]) -> Dict[str, object]:
        descriptor = {
            "quantized": add_tensor_block(f"{block_name}_quantized", payload["quantized"]),
            "scales": add_tensor_block(f"{block_name}_scales", payload["scales"]),
            "reduced_shape": [int(value) for value in payload["reduced_shape"]],
            "original_shape": [int(value) for value in payload["original_shape"]],
        }
        if "keyframe_codec_mode" in payload:
            descriptor["keyframe_codec_mode"] = str(payload["keyframe_codec_mode"])
        if "quant_dtype" in payload:
            descriptor["quant_dtype"] = str(payload["quant_dtype"])
        return descriptor

    global_keyframe_payload = low_payload["global_keyframe"]
    if isinstance(global_keyframe_payload, dict):
        global_keyframe_descriptor = add_quantized_tensor_payload("global_keyframe", global_keyframe_payload)
    else:
        global_keyframe_descriptor = add_tensor_block("global_keyframe", global_keyframe_payload)

    section_anchor_payloads = []
    for section_idx, anchor_payload in enumerate(low_payload["section_anchor_payloads"]):
        section_anchor_payloads.append(
            {
                "quantized": add_tensor_block(f"anchor_quantized_{section_idx:04d}", anchor_payload["quantized"]),
                "scales": add_tensor_block(f"anchor_scales_{section_idx:04d}", anchor_payload["scales"]),
                "reduced_shape": [int(value) for value in anchor_payload["reduced_shape"]],
                "original_shape": [int(value) for value in anchor_payload["original_shape"]],
            }
        )

    section_tail_payloads = []
    for section_idx, tail_payload in enumerate(low_payload["section_tail_payloads"]):
        payload_record = {
            "tail_codec_type": str(tail_payload["tail_codec_type"]),
            "original_shape": [int(value) for value in tail_payload["original_shape"]],
        }
        if "quant_dtype" in tail_payload:
            payload_record["quant_dtype"] = str(tail_payload["quant_dtype"])
        if "entropy_model" in tail_payload:
            payload_record["entropy_model"] = str(tail_payload["entropy_model"])
        if "quantized" in tail_payload:
            payload_record["quantized"] = add_tensor_block(f"tail_quantized_{section_idx:04d}", tail_payload["quantized"])
        if "scales" in tail_payload:
            payload_record["scales"] = add_tensor_block(f"tail_scales_{section_idx:04d}", tail_payload["scales"])
        if "y_string" in tail_payload:
            payload_record["y_string"] = add_bytes_block(
                f"tail_y_string_{section_idx:04d}",
                bytes(tail_payload["y_string"]),
            )
            payload_record["y_shape"] = [int(value) for value in tail_payload["y_shape"]]
        if "z_string" in tail_payload:
            payload_record["z_string"] = add_bytes_block(
                f"tail_z_string_{section_idx:04d}",
                bytes(tail_payload["z_string"]),
            )
            payload_record["z_shape"] = [int(value) for value in tail_payload["z_shape"]]
        if "reduced_shape" in tail_payload:
            payload_record["reduced_shape"] = [int(value) for value in tail_payload["reduced_shape"]]
        if "bitstream_bytes" in tail_payload:
            payload_record["bitstream_bytes"] = int(tail_payload["bitstream_bytes"])
        if "bitstream_bits" in tail_payload:
            payload_record["bitstream_bits"] = int(tail_payload["bitstream_bits"])
        section_tail_payloads.append(payload_record)

    metadata = {
        "format_name": FORMAT_NAME,
        "version": VERSION,
        "entropy_codec": "zstd" if zstd is not None else "zlib",
        "field_strategy": FIELD_STRATEGY,
        "source_payload": _build_source_payload(low_payload, relative_path=relative_path),
        "global_keyframe": global_keyframe_descriptor,
        "section_anchor_payloads": section_anchor_payloads,
        "section_tail_payloads": section_tail_payloads,
        "block_order": block_order,
        "block_lengths": block_lengths,
    }
    metadata_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.parent / f".{output_path.name}.tmp_{os.getpid()}"
    try:
        with temp_path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(VERSION.to_bytes(4, "little", signed=False))
            handle.write(len(metadata_bytes).to_bytes(8, "little", signed=False))
            handle.write(metadata_bytes)
            for payload in block_payloads:
                handle.write(payload)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return {
        "format_name": FORMAT_NAME,
        "entropy_codec": metadata["entropy_codec"],
        "field_strategy": FIELD_STRATEGY,
        "block_lengths": block_lengths,
        "entropy_codec_bytes": int(output_path.stat().st_size),
    }


def read_external_entropy_payload(input_path: Path) -> Dict[str, object]:
    with input_path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Unexpected external entropy magic in {input_path}: {magic!r}")
        version = int.from_bytes(handle.read(4), "little", signed=False)
        if version != VERSION:
            raise ValueError(f"Unsupported external entropy version={version} in {input_path}.")
        metadata_len = int.from_bytes(handle.read(8), "little", signed=False)
        metadata = json.loads(handle.read(metadata_len).decode("utf-8"))
        compressed_blocks = {}
        for block_name in metadata["block_order"]:
            block_len = int(metadata["block_lengths"][block_name])
            compressed_blocks[block_name] = handle.read(block_len)

    def decode_block(block_descriptor: Dict[str, object]) -> bytes:
        block_name = str(block_descriptor["block"])
        compressed = compressed_blocks[block_name]
        return _decompress_bytes(str(block_descriptor["codec"]), compressed)

    source_payload = metadata["source_payload"]
    global_keyframe_metadata = metadata["global_keyframe"]
    if "quantized" in global_keyframe_metadata and "scales" in global_keyframe_metadata:
        global_keyframe = {
            "quantized": _bytes_to_tensor(
                decode_block(global_keyframe_metadata["quantized"]),
                global_keyframe_metadata["quantized"],
            ),
            "scales": _bytes_to_tensor(
                decode_block(global_keyframe_metadata["scales"]),
                global_keyframe_metadata["scales"],
            ),
            "reduced_shape": global_keyframe_metadata["reduced_shape"],
            "original_shape": global_keyframe_metadata["original_shape"],
        }
        if "keyframe_codec_mode" in global_keyframe_metadata:
            global_keyframe["keyframe_codec_mode"] = global_keyframe_metadata["keyframe_codec_mode"]
        if "quant_dtype" in global_keyframe_metadata:
            global_keyframe["quant_dtype"] = global_keyframe_metadata["quant_dtype"]
    else:
        global_keyframe = _bytes_to_tensor(
            decode_block(global_keyframe_metadata),
            global_keyframe_metadata,
        )
    low_payload = {
        "format_version": source_payload["format_version"],
        "input_path": source_payload["input_path"],
        "source_fps": source_payload["source_fps"],
        "source_num_frames": source_payload["source_num_frames"],
        "source_width": source_payload["source_width"],
        "source_height": source_payload["source_height"],
        "chunk_lengths": source_payload["chunk_lengths"],
        "codec_config_v2": source_payload["codec_config_v2"],
        "codec_config": source_payload["codec_config"],
        "tail_codec_type": source_payload["tail_codec_type"],
        "section_ranges": source_payload["section_ranges"],
        "low_codec_bytes": source_payload["low_codec_bytes"],
        "low_bpp": source_payload["low_bpp"],
        "global_keyframe": global_keyframe,
        "section_anchor_payloads": [],
        "section_tail_payloads": [],
    }

    for anchor_payload in metadata["section_anchor_payloads"]:
        low_payload["section_anchor_payloads"].append(
            {
                "quantized": _bytes_to_tensor(
                    decode_block(anchor_payload["quantized"]),
                    anchor_payload["quantized"],
                ),
                "scales": _bytes_to_tensor(
                    decode_block(anchor_payload["scales"]),
                    anchor_payload["scales"],
                ),
                "reduced_shape": anchor_payload["reduced_shape"],
                "original_shape": anchor_payload["original_shape"],
            }
        )

    for tail_payload in metadata["section_tail_payloads"]:
        payload_record = {
            "tail_codec_type": tail_payload["tail_codec_type"],
            "original_shape": tail_payload["original_shape"],
        }
        if "quant_dtype" in tail_payload:
            payload_record["quant_dtype"] = tail_payload["quant_dtype"]
        if "entropy_model" in tail_payload:
            payload_record["entropy_model"] = tail_payload["entropy_model"]
        if "quantized" in tail_payload:
            payload_record["quantized"] = _bytes_to_tensor(
                decode_block(tail_payload["quantized"]),
                tail_payload["quantized"],
            )
        if "scales" in tail_payload:
            payload_record["scales"] = _bytes_to_tensor(
                decode_block(tail_payload["scales"]),
                tail_payload["scales"],
            )
        if "y_string" in tail_payload:
            payload_record["y_string"] = decode_block(tail_payload["y_string"])
            payload_record["y_shape"] = tail_payload["y_shape"]
        if "z_string" in tail_payload:
            payload_record["z_string"] = decode_block(tail_payload["z_string"])
            payload_record["z_shape"] = tail_payload["z_shape"]
        if "reduced_shape" in tail_payload:
            payload_record["reduced_shape"] = tail_payload["reduced_shape"]
        if "bitstream_bytes" in tail_payload:
            payload_record["bitstream_bytes"] = tail_payload["bitstream_bytes"]
        if "bitstream_bits" in tail_payload:
            payload_record["bitstream_bits"] = tail_payload["bitstream_bits"]
        low_payload["section_tail_payloads"].append(payload_record)

    return low_payload
