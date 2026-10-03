"""Parameter-surface stability checks to expose brittle optimized peaks."""

from __future__ import annotations

from statistics import fmean, median, pstdev

from src.modules.strategy.validation.models import ParameterPoint


def analyze_parameter_stability(points: list[ParameterPoint]) -> dict:
    if not points:
        return {
            "point_count": 0,
            "best_objective": None,
            "median_objective": None,
            "objective_cv": None,
            "positive_fraction": None,
            "near_best_fraction": None,
            "stability_score": None,
        }

    objectives = [float(p.objective) for p in points]
    best = max(objectives)
    med = median(objectives)
    mean_abs = fmean(abs(v) for v in objectives)
    cv = None if mean_abs == 0 else pstdev(objectives) / mean_abs
    positive_fraction = sum(1 for v in objectives if v > 0) / len(objectives)

    if best > 0:
        near_best = sum(1 for v in objectives if v >= best * 0.8) / len(objectives)
    else:
        near_best = 0.0

    dispersion_component = 0.0 if cv is None else max(0.0, min(1.0, 1.0 - cv))
    stability_score = (
        0.45 * positive_fraction
        + 0.35 * near_best
        + 0.20 * dispersion_component
    )

    return {
        "point_count": len(points),
        "best_objective": best,
        "median_objective": med,
        "objective_cv": cv,
        "positive_fraction": positive_fraction,
        "near_best_fraction": near_best,
        "stability_score": stability_score,
    }
