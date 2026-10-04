from __future__ import annotations

from dataclasses import dataclass, field

from river import metrics


@dataclass
class OnlinePathEvaluator:
    """Forward-only streaming evaluation using River metrics."""

    candidate_mfe: metrics.MAE = field(default_factory=metrics.MAE)
    candidate_mae: metrics.MAE = field(default_factory=metrics.MAE)
    baseline_mfe: metrics.MAE = field(default_factory=metrics.MAE)
    baseline_mae: metrics.MAE = field(default_factory=metrics.MAE)
    completed: int = 0

    def update(
        self,
        *,
        actual_mfe: float,
        actual_mae: float,
        candidate_mfe: float,
        candidate_mae: float,
        baseline_mfe: float,
        baseline_mae: float,
    ) -> None:
        self.candidate_mfe.update(actual_mfe, candidate_mfe)
        self.candidate_mae.update(actual_mae, candidate_mae)
        self.baseline_mfe.update(actual_mfe, baseline_mfe)
        self.baseline_mae.update(actual_mae, baseline_mae)
        self.completed += 1

    def snapshot(self) -> dict[str, float | int | None]:
        c_mfe = self.candidate_mfe.get()
        c_mae = self.candidate_mae.get()
        b_mfe = self.baseline_mfe.get()
        b_mae = self.baseline_mae.get()

        def ratio(candidate: float, baseline: float) -> float | None:
            return candidate / baseline if baseline > 0 else None

        return {
            "completed": self.completed,
            "candidate_mfe_mae": c_mfe,
            "candidate_mae_mae": c_mae,
            "baseline_mfe_mae": b_mfe,
            "baseline_mae_mae": b_mae,
            "mfe_ratio": ratio(c_mfe, b_mfe),
            "mae_ratio": ratio(c_mae, b_mae),
        }
