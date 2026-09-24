"""Models and shadow inference contract for GTG Event Experience v3.

The models in this module are evidence providers only.  They expose no order,
position-size, stop-loss, target-routing, or risk APIs.

Architectures are dependency-light native PyTorch implementations:
- GRU baseline
- causal TCN
- patch Transformer

UP and DOWN experts are separate model instances trained on separate event
populations.  P(down) is never derived as 1 - P(up).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import math

from src.modules.xau.gtg_event_v3_data import (
    FEATURE_SCHEMA_VERSION,
    LABEL_VERSION,
)

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    F = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


BUNDLE_VERSION = "gtg-event-v3-bundle-v1"
REQUIRED_METADATA_FIELDS = (
    "model_version",
    "feature_schema_version",
    "label_version",
    "dataset_id",
    "training_range",
    "validation_range",
    "oos_range",
    "git_commit",
    "seed",
    "normalization",
    "calibration",
    "architecture",
    "threshold_policy",
)


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError("GTG Event v3 requires optional PyTorch") from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class GTGEventConfig:
    feature_count: int
    sequence_length: int = 96
    hidden_size: int = 64
    embedding_size: int = 48
    dropout: float = 0.10
    patch_length: int = 8
    patch_stride: int = 4
    transformer_heads: int = 4
    transformer_layers: int = 2
    tcn_channels: int = 64
    tcn_layers: int = 4


@dataclass(frozen=True)
class GTGEventPrediction:
    success_probability: float
    adverse_first_probability: float
    expected_mfe_atr: float
    expected_mae_atr: float
    expected_time_to_target_fraction: float
    trend_strength: float
    uncertainty: float
    embedding: list[float]
    model_type: str


if nn is not None:

    class _BaseEncoder(nn.Module):
        model_type = "base"

        def __init__(self, config: GTGEventConfig) -> None:
            super().__init__()
            self.config = config

        def forward(self, x):
            raise NotImplementedError


    class GRUEncoder(_BaseEncoder):
        model_type = "gru"

        def __init__(self, config: GTGEventConfig) -> None:
            super().__init__(config)
            self.norm = nn.LayerNorm(config.feature_count)
            self.gru = nn.GRU(
                input_size=config.feature_count,
                hidden_size=config.hidden_size,
                num_layers=2,
                batch_first=True,
                dropout=config.dropout,
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.embedding_size),
                nn.GELU(),
                nn.LayerNorm(config.embedding_size),
            )

        def forward(self, x):
            x = self.norm(x)
            _out, hidden = self.gru(x)
            return self.proj(hidden[-1])


    class _CausalConv1d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int) -> None:
            super().__init__()
            self.left = (kernel - 1) * dilation
            self.conv = nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                dilation=dilation,
            )

        def forward(self, x):
            return self.conv(F.pad(x, (self.left, 0)))


    class _TCNBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.c1 = _CausalConv1d(channels, channels, 3, dilation)
            self.c2 = _CausalConv1d(channels, channels, 3, dilation)
            self.norm1 = nn.GroupNorm(1, channels)
            self.norm2 = nn.GroupNorm(1, channels)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            x = self.dropout(F.gelu(self.norm1(self.c1(x))))
            x = self.dropout(F.gelu(self.norm2(self.c2(x))))
            return x + residual


    class TCNEncoder(_BaseEncoder):
        model_type = "tcn"

        def __init__(self, config: GTGEventConfig) -> None:
            super().__init__(config)
            self.input_norm = nn.LayerNorm(config.feature_count)
            self.input_proj = nn.Conv1d(
                config.feature_count, config.tcn_channels, kernel_size=1
            )
            self.blocks = nn.ModuleList(
                [
                    _TCNBlock(
                        config.tcn_channels,
                        dilation=2 ** i,
                        dropout=config.dropout,
                    )
                    for i in range(config.tcn_layers)
                ]
            )
            self.proj = nn.Sequential(
                nn.Linear(config.tcn_channels, config.embedding_size),
                nn.GELU(),
                nn.LayerNorm(config.embedding_size),
            )

        def forward(self, x):
            x = self.input_norm(x).transpose(1, 2)
            x = self.input_proj(x)
            for block in self.blocks:
                x = block(x)
            return self.proj(x[:, :, -1])


    class PatchTransformerEncoder(_BaseEncoder):
        model_type = "patch_transformer"

        def __init__(self, config: GTGEventConfig) -> None:
            super().__init__(config)
            if config.embedding_size % config.transformer_heads:
                raise ValueError("embedding_size must be divisible by transformer_heads")
            self.input_norm = nn.LayerNorm(config.feature_count)
            self.patch_proj = nn.Linear(
                config.patch_length * config.feature_count,
                config.embedding_size,
            )
            layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_size,
                nhead=config.transformer_heads,
                dim_feedforward=config.embedding_size * 3,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=config.transformer_layers,
                enable_nested_tensor=False,
            )
            self.out_norm = nn.LayerNorm(config.embedding_size)

        def forward(self, x):
            x = self.input_norm(x)
            patches = x.unfold(
                dimension=1,
                size=self.config.patch_length,
                step=self.config.patch_stride,
            )
            # unfold -> [B, patches, F, patch]; restore patch-major flattening.
            patches = patches.permute(0, 1, 3, 2).contiguous()
            patches = patches.reshape(
                patches.shape[0], patches.shape[1], -1
            )
            tokens = self.patch_proj(patches)
            encoded = self.encoder(tokens)
            return self.out_norm(encoded.mean(dim=1))


    class GTGEventExpert(nn.Module):
        def __init__(self, encoder: _BaseEncoder) -> None:
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config
            self.model_type = encoder.model_type
            d = self.config.embedding_size
            self.success = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
            self.adverse = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
            self.path = nn.Sequential(
                nn.Linear(d, 32),
                nn.GELU(),
                nn.Linear(32, 3),  # mfe, mae, signed trend strength
            )
            self.time = nn.Sequential(nn.Linear(d, 24), nn.GELU(), nn.Linear(24, 1))

        def forward(self, x):
            z = self.encoder(x)
            return {
                "success": self.success(z).squeeze(-1),
                "adverse": self.adverse(z).squeeze(-1),
                "path": self.path(z),
                "time": self.time(z).squeeze(-1),
                "embedding": z,
            }


    class GTGTradeabilityModel(nn.Module):
        """Independent TRADEABLE/ABSTAIN model; it does not predict direction."""

        def __init__(self, encoder: _BaseEncoder) -> None:
            super().__init__()
            self.encoder = encoder
            self.config = encoder.config
            self.model_type = encoder.model_type
            d = self.config.embedding_size
            self.tradeable = nn.Sequential(
                nn.Linear(d, 32),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            z = self.encoder(x)
            return {"tradeable": self.tradeable(z).squeeze(-1), "embedding": z}

else:

    class GTGEventExpert:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            require_torch()

    class GTGTradeabilityModel:  # pragma: no cover
        def __init__(self, *_args, **_kwargs):
            require_torch()


def build_encoder(model_type: str, config: GTGEventConfig):
    require_torch()
    key = str(model_type).lower()
    if key == "gru":
        return GRUEncoder(config)
    if key == "tcn":
        return TCNEncoder(config)
    if key == "patch_transformer":
        return PatchTransformerEncoder(config)
    raise ValueError(f"unknown GTG Event v3 architecture: {model_type}")


def build_expert_model(model_type: str, config: GTGEventConfig):
    return GTGEventExpert(build_encoder(model_type, config))


def build_tradeability_model(model_type: str, config: GTGEventConfig):
    return GTGTradeabilityModel(build_encoder(model_type, config))


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_METADATA_FIELDS if name not in metadata]
    if missing:
        raise ValueError(f"bundle metadata missing required fields: {missing}")
    if metadata["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise ValueError("bundle feature schema version mismatch")
    if metadata["label_version"] != LABEL_VERSION:
        raise ValueError("bundle label version mismatch")


class GTGEventEngine:
    """CPU-safe shadow inference wrapper for one independent event expert."""

    def __init__(
        self,
        model,
        *,
        feature_mean: list[float],
        feature_std: list[float],
        success_temperature: float = 1.0,
        adverse_temperature: float = 1.0,
        device: str = "cpu",
    ) -> None:
        require_torch()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32, device=device)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32, device=device).clamp_min(1e-6)
        self.success_temperature = max(0.05, float(success_temperature))
        self.adverse_temperature = max(0.05, float(adverse_temperature))

    def predict(self, sequence: list[list[float]]) -> GTGEventPrediction:
        require_torch()
        cfg = self.model.config
        if len(sequence) != cfg.sequence_length:
            raise ValueError(f"expected {cfg.sequence_length} rows")
        if any(len(row) != cfg.feature_count for row in sequence):
            raise ValueError("feature width mismatch")
        x = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        if not torch.isfinite(x).all():
            raise ValueError("non-finite GTG v3 features")
        x = (x - self.feature_mean) / self.feature_std
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))
            success = torch.sigmoid(out["success"] / self.success_temperature)[0]
            adverse = torch.sigmoid(out["adverse"] / self.adverse_temperature)[0]
            raw_path = out["path"][0]
            mfe = F.softplus(raw_path[0])
            mae = F.softplus(raw_path[1])
            trend = torch.tanh(raw_path[2]) * 2.0
            time_fraction = torch.sigmoid(out["time"])[0]
            z = out["embedding"][0]
        p = float(success.item())
        return GTGEventPrediction(
            success_probability=p,
            adverse_first_probability=float(adverse.item()),
            expected_mfe_atr=float(mfe.item()),
            expected_mae_atr=float(mae.item()),
            expected_time_to_target_fraction=float(time_fraction.item()),
            trend_strength=float(trend.item()),
            uncertainty=float(4.0 * p * (1.0 - p)),
            embedding=[float(v) for v in z.tolist()],
            model_type=str(self.model.model_type),
        )

    def save_bundle(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any],
    ) -> None:
        require_torch()
        _validate_metadata(metadata)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bundle_version": BUNDLE_VERSION,
                "kind": "expert",
                "model_type": self.model.model_type,
                "config": asdict(self.model.config),
                "state_dict": self.model.state_dict(),
                "feature_mean": self.feature_mean.detach().cpu().tolist(),
                "feature_std": self.feature_std.detach().cpu().tolist(),
                "success_temperature": self.success_temperature,
                "adverse_temperature": self.adverse_temperature,
                "metadata": metadata,
            },
            path,
        )

    @classmethod
    def load_bundle(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        expected_feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> tuple["GTGEventEngine", dict[str, Any]]:
        require_torch()
        payload = torch.load(Path(path), map_location=device)
        if payload.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError("unsupported GTG v3 bundle version")
        if payload.get("kind") != "expert":
            raise ValueError("bundle is not an expert bundle")
        metadata = dict(payload.get("metadata") or {})
        _validate_metadata(metadata)
        if metadata["feature_schema_version"] != expected_feature_schema_version:
            raise ValueError("runtime feature schema does not match bundle")
        cfg = GTGEventConfig(**dict(payload["config"]))
        model = build_expert_model(str(payload["model_type"]), cfg)
        model.load_state_dict(payload["state_dict"])
        engine = cls(
            model,
            feature_mean=list(payload["feature_mean"]),
            feature_std=list(payload["feature_std"]),
            success_temperature=float(payload.get("success_temperature", 1.0)),
            adverse_temperature=float(payload.get("adverse_temperature", 1.0)),
            device=device,
        )
        return engine, metadata

    @classmethod
    def safe_load_bundle(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
        expected_feature_schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> tuple["GTGEventEngine | None", str | None]:
        """Fail-safe loader: callers can fall back to deterministic GTG."""
        try:
            engine, _metadata = cls.load_bundle(
                path,
                device=device,
                expected_feature_schema_version=expected_feature_schema_version,
            )
            return engine, None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"


def save_tradeability_bundle(
    path: str | Path,
    model,
    *,
    feature_mean: list[float],
    feature_std: list[float],
    temperature: float,
    metadata: dict[str, Any],
) -> None:
    require_torch()
    _validate_metadata(metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "bundle_version": BUNDLE_VERSION,
            "kind": "tradeability",
            "model_type": model.model_type,
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "feature_mean": list(feature_mean),
            "feature_std": list(feature_std),
            "temperature": max(0.05, float(temperature)),
            "metadata": metadata,
        },
        path,
    )
