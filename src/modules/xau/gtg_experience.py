"""GTG neural experience engine.

Optional PyTorch module. It is intentionally isolated from the base GTG rules:
the neural layer supplies calibrated probabilities and latent pattern embeddings,
but it must never bypass deterministic risk/invalidation gates.

No future data is used by the encoder. Training/evaluation split policy lives in
benchmarks/gtg_experience_v1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


@dataclass(frozen=True)
class GTGExperienceConfig:
    feature_count: int
    sequence_length: int = 96
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.10
    embedding_size: int = 32
    heads: tuple[str, ...] = (
        "next_move_up",
        "expansion",
        "ma200_touch",
        "ma1000_touch",
        "ma200_rejection",
        "ma1000_rejection",
    )


@dataclass(frozen=True)
class GTGExperiencePrediction:
    probabilities: dict[str, float]
    uncertainty: dict[str, float]
    embedding: list[float]
    analog_summary: dict[str, Any] | None = None


def require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "GTG neural experience requires optional PyTorch; install the "
            "GTG ML environment before loading this module."
        ) from _TORCH_IMPORT_ERROR


if nn is not None:

    class GTGSequenceEncoder(nn.Module):
        """Causal GRU encoder with a compact latent state for analog memory."""

        def __init__(self, config: GTGExperienceConfig) -> None:
            super().__init__()
            self.config = config
            self.input_norm = nn.LayerNorm(config.feature_count)
            self.gru = nn.GRU(
                input_size=config.feature_count,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                batch_first=True,
                dropout=config.dropout if config.num_layers > 1 else 0.0,
            )
            self.projection = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_size, config.embedding_size),
                nn.LayerNorm(config.embedding_size),
            )
            self.heads = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(config.embedding_size, 32),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(32, 1),
                    )
                    for name in config.heads
                }
            )

        def forward(self, x):
            x = self.input_norm(x)
            _, hidden = self.gru(x)
            latent = self.projection(hidden[-1])
            logits = {name: head(latent).squeeze(-1) for name, head in self.heads.items()}
            return logits, latent

else:

    class GTGSequenceEncoder:  # pragma: no cover - import-safe placeholder
        def __init__(self, *_args, **_kwargs) -> None:
            require_torch()


class GTGExperienceEngine:
    """Inference wrapper for probabilities + experience embedding."""

    def __init__(
        self,
        model: GTGSequenceEncoder,
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
        self.feature_mean = torch.tensor(feature_mean, dtype=torch.float32, device=device)
        self.feature_std = torch.tensor(feature_std, dtype=torch.float32, device=device).clamp_min(1e-6)
        self.temperatures = {
            name: max(0.05, float((temperatures or {}).get(name, 1.0)))
            for name in self.model.config.heads
        }

    def predict(self, sequence: list[list[float]]) -> GTGExperiencePrediction:
        require_torch()
        config = self.model.config
        if len(sequence) != config.sequence_length:
            raise ValueError(
                f"expected {config.sequence_length} bars, got {len(sequence)}"
            )
        if any(len(row) != config.feature_count for row in sequence):
            raise ValueError("feature width mismatch")
        x = torch.tensor(sequence, dtype=torch.float32, device=self.device)
        x = (x - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits, latent = self.model(x.unsqueeze(0))
        probabilities = {
            name: float(
                torch.sigmoid(value / self.temperatures.get(name, 1.0))[0].item()
            )
            for name, value in logits.items()
        }
        # Bernoulli entropy-like uncertainty proxy: maximal at p=.5, minimal at 0/1.
        uncertainty = {
            name: float(4.0 * p * (1.0 - p))
            for name, p in probabilities.items()
        }
        return GTGExperiencePrediction(
            probabilities=probabilities,
            uncertainty=uncertainty,
            embedding=[float(v) for v in latent[0].tolist()],
        )

    def save_bundle(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.model.config),
                "state_dict": self.model.state_dict(),
                "feature_mean": self.feature_mean.detach().cpu().tolist(),
                "feature_std": self.feature_std.detach().cpu().tolist(),
                "temperatures": dict(self.temperatures),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load_bundle(cls, path: str | Path, *, device: str = "cpu") -> tuple["GTGExperienceEngine", dict[str, Any]]:
        require_torch()
        payload = torch.load(Path(path), map_location=device)
        raw_config = dict(payload["config"])
        raw_config["heads"] = tuple(raw_config["heads"])
        config = GTGExperienceConfig(**raw_config)
        model = GTGSequenceEncoder(config)
        model.load_state_dict(payload["state_dict"])
        engine = cls(
            model,
            feature_mean=list(payload["feature_mean"]),
            feature_std=list(payload["feature_std"]),
            temperatures=dict(payload.get("temperatures") or {}),
            device=device,
        )
        return engine, dict(payload.get("metadata") or {})


def write_metadata(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
