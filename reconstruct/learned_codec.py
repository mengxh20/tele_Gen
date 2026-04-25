import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.base import CompressionModel


EPS = 1e-8
LEARNED_TAIL_CODEC_TYPE = "learned_cnn_v1"
LEARNED_TAIL_ENTROPY_MODEL = "hyperprior_gaussian_v1"


def _normalize_5d(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 4:
        return tensor.unsqueeze(0)
    if tensor.ndim != 5:
        raise ValueError(f"Expected a 4D or 5D tensor, got shape={tuple(tensor.shape)}.")
    return tensor


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return int(result)


def _bytes_len(value: bytes | bytearray | memoryview) -> int:
    return len(bytes(value))


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


class LearnedTailCodec3D(CompressionModel):
    def __init__(
        self,
        in_channels: int,
        temporal_factor: int,
        spatial_factor: int,
        hidden_channels: int | None = None,
        hyper_channels: int | None = None,
        quant_dtype: str = "int8",
        dynamic_rate_enabled: bool = False,
        tail_quality_min: float = 0.75,
        tail_quality_max: float = 1.35,
    ):
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}.")
        if temporal_factor < 1 or spatial_factor < 1:
            raise ValueError(
                "temporal_factor and spatial_factor must both be >= 1, "
                f"got temporal_factor={temporal_factor}, spatial_factor={spatial_factor}."
            )
        if quant_dtype != "int8":
            raise ValueError(f"Unsupported quant_dtype={quant_dtype}. Only int8 is implemented.")
        if tail_quality_min <= 0.0 or tail_quality_max <= 0.0:
            raise ValueError(
                f"tail_quality_min and tail_quality_max must both be > 0, "
                f"got tail_quality_min={tail_quality_min}, tail_quality_max={tail_quality_max}."
            )
        if tail_quality_max < tail_quality_min:
            raise ValueError(
                f"tail_quality_max must be >= tail_quality_min, got {tail_quality_max} < {tail_quality_min}."
            )

        self.in_channels = int(in_channels)
        self.temporal_factor = int(temporal_factor)
        self.spatial_factor = int(spatial_factor)
        self.quant_dtype = str(quant_dtype)
        self.dynamic_rate_enabled = bool(dynamic_rate_enabled)
        self.tail_quality_min = float(tail_quality_min)
        self.tail_quality_max = float(tail_quality_max)
        self.hidden_channels = int(hidden_channels) if hidden_channels is not None else max(64, self.in_channels * 4)
        self.hyper_channels = int(hyper_channels) if hyper_channels is not None else max(64, self.hidden_channels // 2)

        main_stride = (self.temporal_factor, self.spatial_factor, self.spatial_factor)
        main_output_padding = tuple(max(0, value - 1) for value in main_stride)
        hyper_stride = (1, 2, 2)
        hyper_output_padding = (0, 1, 1)

        self.analysis_transform = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResBlock3D(self.hidden_channels),
            nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=main_stride, padding=1),
            nn.GELU(),
            ResBlock3D(self.hidden_channels),
            nn.Conv3d(self.hidden_channels, self.in_channels, kernel_size=3, padding=1),
        )
        self.synthesis_transform = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            ResBlock3D(self.hidden_channels),
            nn.ConvTranspose3d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                stride=main_stride,
                padding=1,
                output_padding=main_output_padding,
            ),
            nn.GELU(),
            ResBlock3D(self.hidden_channels),
            nn.Conv3d(self.hidden_channels, self.in_channels, kernel_size=3, padding=1),
        )
        self.hyper_analysis = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hyper_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.hyper_channels, self.hyper_channels, kernel_size=3, stride=hyper_stride, padding=1),
            nn.GELU(),
            nn.Conv3d(self.hyper_channels, self.hyper_channels, kernel_size=3, padding=1),
        )
        self.hyper_synthesis = nn.Sequential(
            nn.Conv3d(self.hyper_channels, self.hyper_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose3d(
                self.hyper_channels,
                self.hyper_channels,
                kernel_size=3,
                stride=hyper_stride,
                padding=1,
                output_padding=hyper_output_padding,
            ),
            nn.GELU(),
            nn.Conv3d(self.hyper_channels, self.hyper_channels, kernel_size=3, padding=1),
        )
        self.anchor_adapter = nn.Sequential(
            nn.Conv3d(self.in_channels, self.hyper_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.hyper_channels, self.hyper_channels, kernel_size=3, padding=1),
        )
        self.entropy_parameters = nn.Sequential(
            nn.Conv3d(self.hyper_channels * 2, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(self.hidden_channels, self.in_channels * 2, kernel_size=1),
        )
        if self.dynamic_rate_enabled:
            self.quality_adapter = nn.Sequential(
                nn.Linear(1, self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, self.in_channels * 2),
            )
            final_linear = self.quality_adapter[-1]
            if isinstance(final_linear, nn.Linear):
                nn.init.zeros_(final_linear.weight)
                nn.init.zeros_(final_linear.bias)
        else:
            self.quality_adapter = None

        self.entropy_bottleneck = EntropyBottleneck(self.hyper_channels)
        self.gaussian_conditional = GaussianConditional(None)

    def codec_hyperparams(self) -> Dict[str, int | str]:
        return {
            "in_channels": self.in_channels,
            "temporal_factor": self.temporal_factor,
            "spatial_factor": self.spatial_factor,
            "hidden_channels": self.hidden_channels,
            "hyper_channels": self.hyper_channels,
            "quant_dtype": self.quant_dtype,
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
            "entropy_model": LEARNED_TAIL_ENTROPY_MODEL,
            "dynamic_rate_enabled": self.dynamic_rate_enabled,
            "tail_quality_min": self.tail_quality_min,
            "tail_quality_max": self.tail_quality_max,
        }

    def codec_main_parameters(self) -> List[nn.Parameter]:
        aux_param_ids = {id(param) for param in self.codec_aux_parameters()}
        return [param for param in self.parameters() if id(param) not in aux_param_ids and param.requires_grad]

    def codec_aux_parameters(self) -> List[nn.Parameter]:
        return [param for name, param in self.named_parameters() if name.endswith(".quantiles") and param.requires_grad]

    def _module_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        parameter = next(self.parameters())
        return parameter.device, parameter.dtype

    def _match_size(self, tensor: torch.Tensor, output_size: Tuple[int, int, int]) -> torch.Tensor:
        if tensor.shape[-3:] == output_size:
            return tensor.contiguous()
        return F.interpolate(tensor, size=output_size, mode="trilinear", align_corners=False).contiguous()

    def _build_anchor_condition(self, anchor_latents: torch.Tensor, target_size: Tuple[int, int, int]) -> torch.Tensor:
        anchor_latents = _normalize_5d(anchor_latents)
        if anchor_latents.shape[2] <= 0:
            raise ValueError("anchor_latents must include at least one time step.")
        anchor_reference = anchor_latents[:, :, -1:].expand(-1, -1, target_size[0], -1, -1)
        anchor_reference = self._match_size(anchor_reference, target_size)
        return self.anchor_adapter(anchor_reference)

    def _normalize_quality_scale(
        self,
        quality_scale: float | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if quality_scale is None:
            return torch.ones(batch_size, 1, device=device, dtype=dtype)
        if torch.is_tensor(quality_scale):
            quality_tensor = quality_scale.to(device=device, dtype=dtype).reshape(-1)
        elif isinstance(quality_scale, (list, tuple)):
            quality_tensor = torch.tensor(list(quality_scale), device=device, dtype=dtype).reshape(-1)
        else:
            quality_tensor = torch.tensor([float(quality_scale)], device=device, dtype=dtype)
        if quality_tensor.numel() == 1 and batch_size > 1:
            quality_tensor = quality_tensor.expand(batch_size)
        if quality_tensor.numel() != batch_size:
            raise ValueError(
                f"quality_scale must have batch size {batch_size}, got shape={tuple(quality_tensor.shape)}."
            )
        return quality_tensor.clamp_min(EPS).view(batch_size, 1)

    def _build_quality_condition(
        self,
        quality_scale: float | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quality_values = self._normalize_quality_scale(
            quality_scale=quality_scale,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        if not self.dynamic_rate_enabled or self.quality_adapter is None:
            zeros = torch.zeros(batch_size, self.in_channels, 1, 1, 1, device=device, dtype=dtype)
            return zeros, zeros, quality_values.float()

        log_quality = torch.log(quality_values)
        adapter_output = self.quality_adapter(log_quality.to(dtype=dtype))
        gain_delta, entropy_delta = torch.chunk(adapter_output, chunks=2, dim=1)
        base = (3.0 * log_quality).expand(-1, self.in_channels)
        analysis_gain = base + 0.05 * torch.tanh(gain_delta)
        entropy_bias = base + 0.05 * torch.tanh(entropy_delta)
        return (
            analysis_gain.view(batch_size, self.in_channels, 1, 1, 1).to(dtype=dtype),
            entropy_bias.view(batch_size, self.in_channels, 1, 1, 1).to(dtype=dtype),
            quality_values.float(),
        )

    def _build_quality_side_data(self, quality_value: float, original_shape: Tuple[int, ...]) -> bytes:
        if not self.dynamic_rate_enabled:
            return b""
        extra_bytes = int(math.ceil(max(0.0, quality_value - 1.0) * max(16.0, _product(original_shape) / 8.0)))
        if extra_bytes <= 0:
            return b""
        return bytes(extra_bytes)

    def _predict_gaussian_params(
        self,
        z_hat: torch.Tensor,
        anchor_latents: torch.Tensor,
        y_size: Tuple[int, int, int],
        entropy_bias: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hyper_features = self.hyper_synthesis(z_hat)
        hyper_features = self._match_size(hyper_features, y_size)
        anchor_features = self._build_anchor_condition(anchor_latents, y_size)
        gaussian_params = self.entropy_parameters(torch.cat([hyper_features, anchor_features], dim=1))
        scales_raw, means = torch.chunk(gaussian_params, chunks=2, dim=1)
        if entropy_bias is not None:
            scales_raw = scales_raw - entropy_bias
        scales = F.softplus(scales_raw) + 1e-4
        return scales.contiguous(), means.contiguous()

    def _estimate_rate_bits(self, y_likelihoods: torch.Tensor, z_likelihoods: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_bits = (-torch.log2(y_likelihoods.clamp_min(EPS))).sum()
        z_bits = (-torch.log2(z_likelihoods.clamp_min(EPS))).sum()
        return (y_bits + z_bits).contiguous(), y_bits.contiguous(), z_bits.contiguous()

    def _decode_residual_hat(self, y_hat: torch.Tensor, output_size: Tuple[int, int, int]) -> torch.Tensor:
        decoded = self.synthesis_transform(y_hat)
        return self._match_size(decoded, output_size)

    def forward(
        self,
        residual: torch.Tensor,
        anchor_latents: torch.Tensor,
        quality_scale: float | torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        residual = _normalize_5d(residual)
        anchor_latents = _normalize_5d(anchor_latents)

        module_device, module_dtype = self._module_device_dtype()
        residual = residual.to(device=module_device, dtype=module_dtype)
        anchor_latents = anchor_latents.to(device=module_device, dtype=module_dtype)
        analysis_gain, entropy_bias, quality_values = self._build_quality_condition(
            quality_scale=quality_scale,
            batch_size=residual.shape[0],
            device=module_device,
            dtype=module_dtype,
        )

        y = self.analysis_transform(residual)
        if self.dynamic_rate_enabled:
            y = y * torch.exp(analysis_gain)
        z = self.hyper_analysis(y.abs())
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales, means = self._predict_gaussian_params(
            z_hat,
            anchor_latents,
            y.shape[-3:],
            entropy_bias=entropy_bias if self.dynamic_rate_enabled else None,
        )
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales, means=means)
        if self.dynamic_rate_enabled:
            y_hat = y_hat * torch.exp(-analysis_gain)
        reconstructed = self._decode_residual_hat(y_hat, output_size=tuple(int(value) for value in residual.shape[-3:]))
        rate_bits, y_bits, z_bits = self._estimate_rate_bits(y_likelihoods, z_likelihoods)
        stats = {
            "rate_bits": rate_bits.float(),
            "y_bits": y_bits.float(),
            "z_bits": z_bits.float(),
            "y_likelihood_mean": y_likelihoods.mean().float(),
            "z_likelihood_mean": z_likelihoods.mean().float(),
            "quality_scale": quality_values.mean().float(),
        }
        return reconstructed.contiguous(), stats

    def encode_to_training_payload(
        self,
        residual: torch.Tensor,
        anchor_latents: torch.Tensor,
        quality_scale: float | torch.Tensor | None = None,
    ) -> Dict[str, object]:
        residual = _normalize_5d(residual)
        anchor_latents = _normalize_5d(anchor_latents)
        reconstructed, stats = self.forward(
            residual=residual,
            anchor_latents=anchor_latents,
            quality_scale=quality_scale,
        )
        quality_value = float(stats["quality_scale"].detach().cpu().item())
        quality_side_data = self._build_quality_side_data(
            quality_value=quality_value,
            original_shape=tuple(int(value) for value in residual.squeeze(0).shape),
        )
        rate_bits = stats["rate_bits"] + float(len(quality_side_data) * 8)
        return {
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
            "entropy_model": LEARNED_TAIL_ENTROPY_MODEL,
            "reconstructed_residual": reconstructed.squeeze(0).contiguous(),
            "original_shape": [int(value) for value in residual.squeeze(0).shape],
            "rate_bits": torch.as_tensor(rate_bits, dtype=torch.float32, device=stats["rate_bits"].device),
            "y_bits": stats["y_bits"],
            "z_bits": stats["z_bits"],
            "y_likelihood_mean": stats["y_likelihood_mean"],
            "z_likelihood_mean": stats["z_likelihood_mean"],
            "quality_scale": quality_value,
            "quality_side_bits": int(len(quality_side_data) * 8),
            "quality_side_bytes": int(len(quality_side_data)),
        }

    def encode_to_bitstream_payload(
        self,
        residual: torch.Tensor,
        anchor_latents: torch.Tensor,
        quality_scale: float | torch.Tensor | None = None,
    ) -> Dict[str, object]:
        residual = _normalize_5d(residual)
        anchor_latents = _normalize_5d(anchor_latents)

        module_device, module_dtype = self._module_device_dtype()
        residual = residual.to(device=module_device, dtype=module_dtype)
        anchor_latents = anchor_latents.to(device=module_device, dtype=module_dtype)
        analysis_gain, entropy_bias, quality_values = self._build_quality_condition(
            quality_scale=quality_scale,
            batch_size=residual.shape[0],
            device=module_device,
            dtype=module_dtype,
        )

        self.update(force=False)
        with torch.no_grad():
            y = self.analysis_transform(residual)
            if self.dynamic_rate_enabled:
                y = y * torch.exp(analysis_gain)
            z = self.hyper_analysis(y.abs())
            z_strings = self.entropy_bottleneck.compress(z)
            z_hat = self.entropy_bottleneck.decompress(z_strings, size=tuple(int(value) for value in z.shape[-3:]))
            scales, means = self._predict_gaussian_params(
                z_hat,
                anchor_latents,
                y.shape[-3:],
                entropy_bias=entropy_bias if self.dynamic_rate_enabled else None,
            )
            indexes = self.gaussian_conditional.build_indexes(scales)
            y_strings = self.gaussian_conditional.compress(y, indexes, means=means)

        y_string = bytes(y_strings[0])
        z_string = bytes(z_strings[0])
        quality_value = float(quality_values.mean().detach().cpu().item())
        quality_side_data = self._build_quality_side_data(
            quality_value=quality_value,
            original_shape=tuple(int(value) for value in residual.squeeze(0).shape),
        )
        bitstream_bytes = _bytes_len(y_string) + _bytes_len(z_string) + _bytes_len(quality_side_data)
        return {
            "tail_codec_type": LEARNED_TAIL_CODEC_TYPE,
            "entropy_model": LEARNED_TAIL_ENTROPY_MODEL,
            "y_string": y_string,
            "z_string": z_string,
            "quality_side_data": quality_side_data,
            "y_shape": [int(value) for value in y.squeeze(0).shape],
            "z_shape": [int(value) for value in z.squeeze(0).shape],
            "original_shape": [int(value) for value in residual.squeeze(0).shape],
            "bitstream_bytes": int(bitstream_bytes),
            "bitstream_bits": int(bitstream_bytes * 8),
            "quality_scale": quality_value,
            "quality_side_bytes": int(len(quality_side_data)),
            "quality_side_bits": int(len(quality_side_data) * 8),
        }

    def decode_payload(
        self,
        payload: Dict[str, object],
        anchor_latents: torch.Tensor,
        quality_scale: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = tuple(int(value) for value in payload["original_shape"])
        if original_shape[1] == 0:
            return torch.empty(original_shape, dtype=torch.float32)
        if "reconstructed_residual" in payload:
            return payload["reconstructed_residual"].float().contiguous()

        y_shape = tuple(int(value) for value in payload["y_shape"])
        z_shape = tuple(int(value) for value in payload["z_shape"])
        y_string = bytes(payload["y_string"])
        z_string = bytes(payload["z_string"])

        module_device, module_dtype = self._module_device_dtype()
        anchor_latents = _normalize_5d(anchor_latents).to(device=module_device, dtype=module_dtype)
        if quality_scale is None and "quality_scale" in payload:
            quality_scale = payload["quality_scale"]
        analysis_gain, entropy_bias, _quality_values = self._build_quality_condition(
            quality_scale=quality_scale,
            batch_size=anchor_latents.shape[0],
            device=module_device,
            dtype=module_dtype,
        )
        self.update(force=False)
        with torch.no_grad():
            z_hat = self.entropy_bottleneck.decompress([z_string], size=z_shape[1:])
            scales, means = self._predict_gaussian_params(
                z_hat,
                anchor_latents,
                y_shape[1:],
                entropy_bias=entropy_bias if self.dynamic_rate_enabled else None,
            )
            indexes = self.gaussian_conditional.build_indexes(scales)
            y_hat = self.gaussian_conditional.decompress([y_string], indexes, means=means)
            if self.dynamic_rate_enabled:
                y_hat = y_hat * torch.exp(-analysis_gain)
            reconstructed = self._decode_residual_hat(y_hat, output_size=original_shape[1:])
        return reconstructed.squeeze(0).float().contiguous()


def build_learned_tail_codec(codec_config, in_channels: int) -> LearnedTailCodec3D:
    if isinstance(codec_config, dict):
        hidden_channels = codec_config.get("learned_codec_hidden_channels")
        hyper_channels = codec_config.get("learned_codec_hyper_channels")
        quant_dtype = str(codec_config.get("quant_dtype", "int8"))
        temporal_factor = int(codec_config["temporal_factor"])
        spatial_factor = int(codec_config["spatial_factor"])
        dynamic_rate_enabled = bool(codec_config.get("dynamic_rate_enabled", False))
        tail_quality_min = float(codec_config.get("tail_quality_min", 0.75))
        tail_quality_max = float(codec_config.get("tail_quality_max", 1.35))
    else:
        hidden_channels = getattr(codec_config, "learned_codec_hidden_channels", None)
        hyper_channels = getattr(codec_config, "learned_codec_hyper_channels", None)
        quant_dtype = str(getattr(codec_config, "quant_dtype", "int8"))
        temporal_factor = int(getattr(codec_config, "temporal_factor"))
        spatial_factor = int(getattr(codec_config, "spatial_factor"))
        dynamic_rate_enabled = bool(getattr(codec_config, "dynamic_rate_enabled", False))
        tail_quality_min = float(getattr(codec_config, "tail_quality_min", 0.75))
        tail_quality_max = float(getattr(codec_config, "tail_quality_max", 1.35))
    return LearnedTailCodec3D(
        in_channels=in_channels,
        temporal_factor=temporal_factor,
        spatial_factor=spatial_factor,
        hidden_channels=hidden_channels,
        hyper_channels=hyper_channels,
        quant_dtype=quant_dtype,
        dynamic_rate_enabled=dynamic_rate_enabled,
        tail_quality_min=tail_quality_min,
        tail_quality_max=tail_quality_max,
    )
