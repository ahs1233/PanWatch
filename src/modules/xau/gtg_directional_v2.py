"""GTG Directional Experience v2.

Causal, optional PyTorch models for learning how gold moves UP, DOWN, or remains
NEUTRAL over volatility-scaled horizons.  The module is isolated from execution
and risk logic: it provides calibrated directional/path evidence only.

Architectures are native and dependency-light on purpose:
- GRU baseline
- Patch Transformer inspired by PatchTST-style temporal patching
- TimesNetLite inspired by adaptive multi-period 2D temporal variation modeling

No model is allowed to become decision authority merely because training succeeds.
Promotion is controlled by OOS benchmarks and deterministic GTG gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import math

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


DIRECTION_CLASSES = ("down", "neutral", "up")


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "GTG Directional Experience requires optional PyTorch."
        ) from _TORCH_IMPORT_ERROR


@dataclass(frozen=True)
class GTGDirectionalConfig:
    feature_count: int
    sequence_length: int = 96
    horizons_minutes: tuple[int, ...] = (30, 60, 120, 240)
    hidden_size: int = 64
    embedding_size: int = 48
    dropout: float = 0.10
    patch_length: int = 8
    patch_stride: int = 4
    transformer_heads: int = 4
    transformer_layers: int = 2
    timesnet_top_k: int = 3
    timesnet_blocks: int = 2


@dataclass(frozen=True)
class GTGDirectionalPrediction:
    direction_probabilities: dict[str, dict[str, float]]
    path: dict[str, float]
    embedding: list[float]
    model_type: str
    uncertainty: dict[str, float]


if nn is not None:

    class _DirectionalHeads(nn.Module):
        def __init__(self, config: GTGDirectionalConfig) -> None:
            super().__init__()
            self.config = config
            self.direction = nn.ModuleDict({
                str(h): nn.Sequential(
                    nn.Linear(config.embedding_size, 32),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(32, 3),
                )
                for h in config.horizons_minutes
            })
            # up_mfe_atr, down_mfe_atr, trend_tstat_scaled
            self.path = nn.Sequential(
                nn.Linear(config.embedding_size, 32),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(32, 3),
            )
            # normalized first-touch times for up/down for every horizon.
            self.time = nn.Sequential(
                nn.Linear(config.embedding_size, 32),
                nn.GELU(),
                nn.Linear(32, 2 * len(config.horizons_minutes)),
            )

        def forward(self, latent):
            return {
                "direction": {h: head(latent) for h, head in self.direction.items()},
                "path": self.path(latent),
                "time": self.time(latent),
            }


    class _BaseDirectionalModel(nn.Module):
        model_type = "base"

        def __init__(self, config: GTGDirectionalConfig) -> None:
            super().__init__()
            self.config = config
            self.heads = _DirectionalHeads(config)

        def encode(self, x):
            raise NotImplementedError

        def forward(self, x):
            latent = self.encode(x)
            outputs = self.heads(latent)
            outputs["embedding"] = latent
            return outputs


    class DirectionalGRU(_BaseDirectionalModel):
        model_type = "gru"

        def __init__(self, config: GTGDirectionalConfig) -> None:
            super().__init__(config)
            self.input_norm = nn.LayerNorm(config.feature_count)
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

        def encode(self, x):
            _, hidden = self.gru(self.input_norm(x))
            return self.proj(hidden[-1])


    class DirectionalPatchTransformer(_BaseDirectionalModel):
        model_type = "patch_transformer"

        def __init__(self, config: GTGDirectionalConfig) -> None:
            super().__init__(config)
            d_model = config.hidden_size
            self.input_norm = nn.LayerNorm(config.feature_count)
            self.patch = nn.Conv1d(
                config.feature_count,
                d_model,
                kernel_size=config.patch_length,
                stride=config.patch_stride,
            )
            patch_count = 1 + max(
                0,
                (config.sequence_length - config.patch_length) // config.patch_stride,
            )
            self.pos = nn.Parameter(torch.zeros(1, patch_count, d_model))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=config.transformer_heads,
                dim_feedforward=d_model * 4,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.transformer_layers,
            )
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, config.embedding_size),
                nn.GELU(),
                nn.LayerNorm(config.embedding_size),
            )

        def encode(self, x):
            x = self.input_norm(x).transpose(1, 2)
            patches = self.patch(x).transpose(1, 2)
            patches = patches + self.pos[:, : patches.size(1)]
            encoded = self.encoder(patches)
            return self.proj(encoded.mean(dim=1))


    class _PeriodBlock(nn.Module):
        def __init__(self, d_model: int, top_k: int, dropout: float) -> None:
            super().__init__()
            self.top_k = top_k
            self.conv = nn.Sequential(
                nn.Conv2d(d_model, d_model, kernel_size=(1, 3), padding=(0, 1)),
                nn.GELU(),
                nn.Conv2d(d_model, d_model, kernel_size=(3, 1), padding=(1, 0)),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            # x: [B, T, D]. Frequency selection is based only on the causal input.
            b, t, d = x.shape
            spectrum = torch.fft.rfft(x.float(), dim=1)
            amplitude = spectrum.abs().mean(dim=(0, 2))
            if amplitude.numel() <= 1:
                return self.norm(x)
            amplitude = amplitude.clone()
            amplitude[0] = 0.0
            k = min(self.top_k, max(1, amplitude.numel() - 1))
            freq_idx = torch.topk(amplitude, k=k).indices.clamp_min(1)
            raw_weights = amplitude[freq_idx]
            weights = torch.softmax(raw_weights, dim=0)

            mixed = torch.zeros_like(x)
            for weight, freq in zip(weights, freq_idx):
                f = max(1, int(freq.item()))
                period = max(2, min(t, int(round(t / f))))
                padded_t = int(math.ceil(t / period) * period)
                if padded_t > t:
                    pad = torch.zeros(
                        b, padded_t - t, d, dtype=x.dtype, device=x.device
                    )
                    z = torch.cat([x, pad], dim=1)
                else:
                    z = x
                # [B, periods, period, D] -> [B, D, periods, period]
                z = z.reshape(b, padded_t // period, period, d).permute(0, 3, 1, 2)
                z = self.conv(z)
                z = z.permute(0, 2, 3, 1).reshape(b, padded_t, d)[:, :t]
                mixed = mixed + weight.to(x.dtype) * z
            return self.norm(x + mixed)


    class DirectionalTimesNetLite(_BaseDirectionalModel):
        model_type = "timesnet_lite"

        def __init__(self, config: GTGDirectionalConfig) -> None:
            super().__init__(config)
            d_model = config.hidden_size
            self.input_norm = nn.LayerNorm(config.feature_count)
            self.input_proj = nn.Linear(config.feature_count, d_model)
            self.blocks = nn.ModuleList([
                _PeriodBlock(d_model, config.timesnet_top_k, config.dropout)
                for _ in range(config.timesnet_blocks)
            ])
            self.proj = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, config.embedding_size),
                nn.GELU(),
                nn.LayerNorm(config.embedding_size),
            )

        def encode(self, x):
            z = self.input_proj(self.input_norm(x))
            for block in self.blocks:
                z = block(z)
            return self.proj(z.mean(dim=1))

else:

    class DirectionalGRU:  # pragma: no cover
        def __init__(self, *_a, **_k):
            require_torch()

    class DirectionalPatchTransformer(DirectionalGRU):
        pass

    class DirectionalTimesNetLite(DirectionalGRU):
        pass


MODEL_REGISTRY = {
    "gru": DirectionalGRU,
    "patch_transformer": DirectionalPatchTransformer,
    "timesnet_lite": DirectionalTimesNetLite,
}


def build_directional_model(model_type: str, config: GTGDirectionalConfig):
    require_torch()
    try:
        cls = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(
            f"unknown GTG directional model {model_type!r}; "
            f"expected one of {sorted(MODEL_REGISTRY)}"
        ) from exc
    return cls(config)


class GTGDirectionalEngine:
    """Calibrated inference wrapper.

    The engine exposes evidence only.  It has no order-routing, sizing, or risk
    authority, which keeps model lifecycle failures isolated from execution.
    """

    def __init__(
        self,
        model,
        *,
        feature_mean: list[float],
        feature_std: list[float],
        temperatures: dict[str, float] | None = None,
        device: str = "cpu",
    ) -> None:
        require_torch()
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.mean = torch.tensor(feature_mean, dtype=torch.float32, device=device)
        self.std = torch.tensor(feature_std, dtype=torch.float32, device=device).clamp_min(1e-6)
        self.temperatures = {
            str(k): max(0.05, float(v))
            for k, v in (temperatures or {}).items()
        }

    def predict(self, sequence: list[list[float]]) -> GTGDirectionalPrediction:
        cfg = self.model.config
        if len(sequence) != cfg.sequence_length:
            raise ValueError(
                f"expected {cfg.sequence_length} bars, got {len(sequence)}"
            )
        if any(len(row) != cfg.feature_count for row in sequence):
            raise ValueError("feature width mismatch")
        x = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        x = (x - self.mean) / self.std
        with torch.no_grad():
            out = self.model(x.unsqueeze(0))

        probs: dict[str, dict[str, float]] = {}
        uncertainty: dict[str, float] = {}
        for h in cfg.horizons_minutes:
            key = str(h)
            temperature = self.temperatures.get(key, 1.0)
            p = torch.softmax(out["direction"][key] / temperature, dim=-1)[0]
            vals = [float(v) for v in p.tolist()]
            probs[key] = dict(zip(DIRECTION_CLASSES, vals))
            entropy = -sum(v * math.log(max(v, 1e-12)) for v in vals) / math.log(3.0)
            uncertainty[key] = float(entropy)

        path_raw = out["path"][0].tolist()
        path = {
            "up_mfe_atr": max(0.0, float(path_raw[0])),
            "down_mfe_atr": max(0.0, float(path_raw[1])),
            "trend_tstat_scaled": float(path_raw[2]),
        }
        return GTGDirectionalPrediction(
            direction_probabilities=probs,
            path=path,
            embedding=[float(v) for v in out["embedding"][0].tolist()],
            model_type=self.model.model_type,
            uncertainty=uncertainty,
        )

    def save_bundle(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_type": self.model.model_type,
            "config": asdict(self.model.config),
            "state_dict": self.model.state_dict(),
            "feature_mean": self.mean.detach().cpu().tolist(),
            "feature_std": self.std.detach().cpu().tolist(),
            "temperatures": dict(self.temperatures),
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load_bundle(cls, path: str | Path, *, device: str = "cpu"):
        require_torch()
        payload = torch.load(Path(path), map_location=device)
        raw = dict(payload["config"])
        raw["horizons_minutes"] = tuple(raw["horizons_minutes"])
        config = GTGDirectionalConfig(**raw)
        model = build_directional_model(payload["model_type"], config)
        model.load_state_dict(payload["state_dict"])
        engine = cls(
            model,
            feature_mean=list(payload["feature_mean"]),
            feature_std=list(payload["feature_std"]),
            temperatures=dict(payload.get("temperatures") or {}),
            device=device,
        )
        return engine, dict(payload.get("metadata") or {})
