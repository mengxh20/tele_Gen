import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8
LEARNED_TAIL_CODEC_TYPE = "learned_cnn_v1"


def _normalize_5d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 4D or 5D tensor, got shape={tuple(tensor.shape)}.")
    return tensor


def _quant_reduce_dims(tensor: torch.Tensor) -> Tuple[int, ...]:
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 5D tensor, got shape={tuple(tensor.shape)}.")
    return (0, 2, 3, 4)


def _symmetric_quantize_per_channel_5d(tensor: torch.Tensor, quant_dtype: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if quant_dtype != "int8":
        raise ValueError(f"Unsupported quant_dtype={quant_dtype}. Only int8 is implemented.")
    tensor = _normalize_5d(tensor).float()
    scales = tensor.abs().amax(dim=_quant_reduce_dims(tensor), keepdim=True).clamp_min(EPS) / 127.0
    quantized = torch.round(tensor / scales).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), scales.contiguous()


def _symmetric_dequantize_per_channel_5d(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    quantized = _normalize_5d(quantized).float()
    scales = _normalize_5d(scales)
    return (quantized * scales.float()).contiguous()


def _fake_quantize_ste_per_channel_5d(tensor: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    quantized, scales = _symmetric_quantize_per_channel_5d(tensor, quant_dtype)
    dequantized = _symmetric_dequantize_per_channel_5d(quantized, scales)
    dequantized = dequantized.to(device=tensor.device, dtype=tensor.dtype)
    return tensor + (dequantized - tensor).detach()


def _squeeze_payload_scale(scales: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_5d(scales)
    return normalized.squeeze(0).squeeze(-1).squeeze(-1).squeeze(-1).float().contiguous()


def _unsqueeze_payload_scale(scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim != 1:
        raise ValueError(f"Expected scales with shape [C], got {tuple(scales.shape)}.")
    return scales.view(1, -1, 1, 1, 1).float().contiguous()


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + residual)


class LearnedTailCodec3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temporal_factor: int,
        spatial_factor: int,
        hidden_channels: int | None = None,
        quant_dtype: str = "int8",
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if temporal_factor < 1 or spatial_factor < 1:
            raise ValueError(
                "temporal_factor and spatial_factor must both be >= 1, "
                f"got temporal_factor={temporal_factor}, spatial_factor={spatial_factor}."
            )
        self.in_channels = int(in_channels)
        self.temporal_factor = int(temporal_factor)
        self.spatial_factor = int(spatial_factor)
        self.quant_dtype = str(quant_dtype)
        hidden = int(hidden_channels) if hidden_channels is not None else max(64, self.in_channels * 4)
        stride = (self.temporal_factor, self.spatial_factor, self.spatial_factor)
        output_padding = tuple(max(0, value - 1) for value in stride)

        self.encoder = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            ResBlock3D(hidden),
            nn.Conv3d(hidden, hidden, kernel_size=3, stride=stride, padding=1),
            nn.GELU(),
            ResBlock3D(hidden),
            nn.Conv3d(hidden, self.in_channels, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv3d(self.in_channels, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            ResBlock3D(hidden),
            nn.ConvTranspose3d(
                hidden,
                hidden,
                kernel_size=3,
                stride=stride,
                padding=1,
                output_padding=output_padding,
            ),
            nn.GELU(),
            ResBlock3D(hidden),
            nn.Conv3d(hidden, self.in_channels, kernel_size=3, padding=1),
        )

    def codec_hyperparams(self) -> Dict[str, int | str]:
        return {
            "in_channels": self.in_channels,
            "temporal_factor": self.temporal_factor,
            "spatial_factor": self.spatial_factor,
            "hidden_channels": int(self.encoder[0].out_channels),
            "quant_dtype": self.quant_dtype,
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
        }

    def _module_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def encode(self, residual: torch.Tensor) -> torch.Tensor:
        module_device, module_dtype = self._module_device_dtype()
        residual = _normalize_5d(residual).to(device=module_device, dtype=module_dtype)
        return self.encoder(residual).contiguous()

    def decode(self, bottleneck: torch.Tensor, output_size: Tuple[int, int, int]) -> torch.Tensor:
        module_device, module_dtype = self._module_device_dtype()
        bottleneck = _normalize_5d(bottleneck).to(device=module_device, dtype=module_dtype)
        decoded = self.decoder(bottleneck)
        if decoded.shape[-3:] != output_size:
            decoded = F.interpolate(decoded, size=output_size, mode="trilinear", align_corners=False)
        return decoded.contiguous()

    def forward(self, residual: torch.Tensor, use_ste_quant: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = _normalize_5d(residual)
        bottleneck = self.encode(residual)
        if use_ste_quant:
            bottleneck = _fake_quantize_ste_per_channel_5d(bottleneck, self.quant_dtype)
        reconstructed = self.decode(bottleneck, output_size=tuple(int(value) for value in residual.shape[-3:]))
        return reconstructed, bottleneck

    def encode_to_ste_payload(self, residual: torch.Tensor) -> Dict[str, object]:
        residual = _normalize_5d(residual)
        bottleneck = self.encode(residual)
        fake_quantized = _fake_quantize_ste_per_channel_5d(bottleneck, self.quant_dtype)
        return {
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
            "quant_dtype": self.quant_dtype,
            "ste_bottleneck": fake_quantized.squeeze(0).contiguous(),
            "reduced_shape": [int(value) for value in fake_quantized.squeeze(0).shape],
            "original_shape": [int(value) for value in residual.squeeze(0).shape],
        }

    def encode_to_quantized_payload(self, residual: torch.Tensor) -> Dict[str, object]:
        residual = _normalize_5d(residual)
        bottleneck = self.encode(residual)
        quantized, scales = _symmetric_quantize_per_channel_5d(bottleneck, self.quant_dtype)
        quantized_payload = quantized.squeeze(0).cpu().contiguous()
        scale_payload = _squeeze_payload_scale(scales).cpu().contiguous()
        return {
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
            "quant_dtype": self.quant_dtype,
            "quantized": quantized_payload,
            "scales": scale_payload,
            "reduced_shape": [int(value) for value in quantized_payload.shape],
            "original_shape": [int(value) for value in residual.squeeze(0).shape],
        }

    def decode_payload(self, payload: Dict[str, object]) -> torch.Tensor:
        original_shape = tuple(int(value) for value in payload["original_shape"])
        if original_shape[1] == 0:
            return torch.empty(original_shape, dtype=torch.float32)

        module_device, module_dtype = self._module_device_dtype()
        if "ste_bottleneck" in payload:
            bottleneck = _normalize_5d(payload["ste_bottleneck"]).to(device=module_device, dtype=module_dtype)
        else:
            quantized = payload["quantized"]
            if isinstance(quantized, torch.Tensor) and quantized.numel() == 0:
                reduced_shape = tuple(int(value) for value in payload["reduced_shape"])
                bottleneck = torch.empty((1, *reduced_shape), device=module_device, dtype=module_dtype)
            else:
                bottleneck = _symmetric_dequantize_per_channel_5d(
                    quantized=_normalize_5d(payload["quantized"]).to(device=module_device),
                    scales=_unsqueeze_payload_scale(payload["scales"]).to(device=module_device),
                ).to(dtype=module_dtype)
        decoded = self.decode(bottleneck, output_size=original_shape[1:])
        return decoded.squeeze(0).contiguous()


def build_learned_tail_codec(codec_config, in_channels: int) -> LearnedTailCodec3D:
    hidden_channels = None
    if isinstance(codec_config, dict):
        hidden_channels = codec_config.get("learned_codec_hidden_channels")
        quant_dtype = str(codec_config.get("quant_dtype", "int8"))
        temporal_factor = int(codec_config["temporal_factor"])
        spatial_factor = int(codec_config["spatial_factor"])
    else:
        hidden_channels = getattr(codec_config, "learned_codec_hidden_channels", None)
        quant_dtype = str(getattr(codec_config, "quant_dtype", "int8"))
        temporal_factor = int(getattr(codec_config, "temporal_factor"))
        spatial_factor = int(getattr(codec_config, "spatial_factor"))
    return LearnedTailCodec3D(
        in_channels=in_channels,
        temporal_factor=temporal_factor,
        spatial_factor=spatial_factor,
        hidden_channels=hidden_channels,
        quant_dtype=quant_dtype,
    )
