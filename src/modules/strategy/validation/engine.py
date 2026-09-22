"""Engine-neutral validation orchestrator for trading hypotheses."""

from __future__ import annotations

from dataclasses import asdict

from src.modules.strategy.validation.metrics import compute_performance
from src.modules.strategy.validation.models import (
    ExperimentSpec,
    GateResult,
    ParameterPoint,
    TradeRecord,
    ValidationReport,
    ValidationVerdict,
    WalkForwardFoldResult,
)
from src.modules.strategy.validation.monte_carlo import bootstrap_trade_sequences
from src.modules.strategy.validation.segmentation import segment_performance
from src.modules.strategy.validation.splits import chronological_split
from src.modules.strategy.validation.stability import analyze_parameter_stability
from src.modules.strategy.validation.stress import cost_stress


def _walk_forward_summary(
    results: list[WalkForwardFoldResult] | None,
) -> dict:
    """Summarize externally produced, leakage-guarded walk-forward folds."""

    if not results:
        return {
            "fold_count": 0,
            "positive_test_fraction": None,
            "folds": [],
            "provenance": "missing",
        }

    rows: list[dict] = []
    ordered = sorted(results, key=lambda item: item.fold_index)
    for fold in ordered:
        test = compute_performance(list(fold.test_trades))
        rows.append(
            {
                "fold": fold.fold_index,
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "selected_at": fold.selected_at.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "selected_strategy_fingerprint": fold.selected_strategy_fingerprint,
                "selected_parameters": dict(fold.selected_parameters),
                "test": asdict(test),
                "test_positive": bool(test.expectancy is not None and test.expectancy > 0),
            }
        )

    fraction = sum(1 for row in rows if row["test_positive"]) / len(rows)
    return {
        "fold_count": len(rows),
        "positive_test_fraction": fraction,
        "folds": rows,
        "provenance": "train_select_future_test",
    }


def _gates(
    spec: ExperimentSpec,
    oos,
    walk_forward: dict,
    monte_carlo: dict,
    stability: dict,
    *,
    parameter_surface_supplied: bool,
    parameter_surface_is_in_sample_only: bool,
) -> tuple[GateResult, ...]:
    policy = spec.policy
    gates: list[GateResult] = []

    gates.append(
        GateResult(
            "strategy_frozen_before_oos",
            spec.strategy_frozen_before_oos,
            spec.strategy_frozen_before_oos,
            True,
            "Prevents tuning on information from the holdout period.",
        )
    )
    gates.append(
        GateResult(
            "oos_trade_count",
            oos.trade_count >= policy.min_oos_trades,
            oos.trade_count,
            policy.min_oos_trades,
        )
    )
    gates.append(
        GateResult(
            "oos_expectancy",
            None if oos.expectancy is None else oos.expectancy > policy.min_oos_expectancy,
            oos.expectancy,
            policy.min_oos_expectancy,
        )
    )
    pf = oos.profit_factor
    gates.append(
        GateResult(
            "oos_profit_factor",
            None if pf is None else pf >= policy.min_oos_profit_factor,
            pf,
            policy.min_oos_profit_factor,
        )
    )

    if policy.max_oos_drawdown is not None:
        gates.append(
            GateResult(
                "oos_max_drawdown",
                oos.max_drawdown <= policy.max_oos_drawdown,
                oos.max_drawdown,
                policy.max_oos_drawdown,
            )
        )

    wf_fraction = walk_forward.get("positive_test_fraction")
    gates.append(
        GateResult(
            "walk_forward_positive_fraction",
            None if wf_fraction is None else wf_fraction >= policy.min_walk_forward_positive_fraction,
            wf_fraction,
            policy.min_walk_forward_positive_fraction,
            "Requires real train->selection->future-test fold evidence.",
        )
    )

    stability_score = stability.get("stability_score")
    if not parameter_surface_supplied:
        stability_passed = None
        stability_detail = "No parameter surface supplied."
    elif not parameter_surface_is_in_sample_only:
        stability_passed = False
        stability_detail = "Parameter surface touched non-IS data; stability evidence is contaminated."
    else:
        stability_passed = (
            None
            if stability_score is None
            else stability_score >= policy.min_parameter_stability_score
        )
        stability_detail = "Parameter surface declared in-sample-only."

    gates.append(
        GateResult(
            "parameter_stability",
            stability_passed,
            stability_score,
            policy.min_parameter_stability_score,
            stability_detail,
        )
    )

    loss_probability = monte_carlo.get("loss_probability")
    gates.append(
        GateResult(
            "monte_carlo_loss_probability",
            None if loss_probability is None else loss_probability <= policy.max_monte_carlo_loss_probability,
            loss_probability,
            policy.max_monte_carlo_loss_probability,
        )
    )
    return tuple(gates)


def validate_experiment(
    *,
    spec: ExperimentSpec,
    trades: list[TradeRecord],
    parameter_surface: list[ParameterPoint] | None = None,
    parameter_surface_is_in_sample_only: bool = False,
    walk_forward_results: list[WalkForwardFoldResult] | None = None,
    cost_per_trade_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50),
) -> ValidationReport:
    ordered = sorted(trades, key=lambda t: (t.closed_at, t.trade_id))
    ins, oos_trades = chronological_split(ordered, spec.in_sample_fraction)

    in_sample = compute_performance(ins)
    out_of_sample = compute_performance(oos_trades)
    walk_forward = _walk_forward_summary(walk_forward_results)
    monte_carlo = bootstrap_trade_sequences(
        oos_trades,
        iterations=spec.monte_carlo_iterations,
        seed=spec.monte_carlo_seed,
    )
    stability = analyze_parameter_stability(parameter_surface or [])
    segmentation = segment_performance(oos_trades)
    stress = cost_stress(oos_trades, cost_per_trade_levels)

    gates = _gates(
        spec,
        out_of_sample,
        walk_forward,
        monte_carlo,
        stability,
        parameter_surface_supplied=bool(parameter_surface),
        parameter_surface_is_in_sample_only=parameter_surface_is_in_sample_only,
    )
    known = [g for g in gates if g.passed is not None]
    unknown = [g for g in gates if g.passed is None]

    limitations: list[str] = []
    if unknown:
        limitations.append(
            "Some evidence gates are unavailable: "
            + ", ".join(g.name for g in unknown)
        )
    if not walk_forward_results:
        limitations.append(
            "No true walk-forward fold results supplied; rolling slices are not treated as walk-forward evidence."
        )
    if not parameter_surface:
        limitations.append("No parameter surface supplied; optimization stability is unproven.")
    elif not parameter_surface_is_in_sample_only:
        limitations.append("Parameter surface is not proven to be in-sample-only.")
    if not spec.strategy_frozen_before_oos:
        limitations.append("Strategy was not declared frozen before OOS evaluation.")
    if not any(t.mae is not None for t in oos_trades):
        limitations.append("OOS MAE is unavailable.")
    if not any(t.mfe is not None for t in oos_trades):
        limitations.append("OOS MFE is unavailable.")

    if any(g.passed is False for g in known):
        verdict = ValidationVerdict.FAIL
    elif unknown:
        verdict = ValidationVerdict.INSUFFICIENT_EVIDENCE
    else:
        verdict = ValidationVerdict.PASS

    return ValidationReport(
        experiment_id=spec.experiment_id,
        strategy_fingerprint=spec.strategy_fingerprint,
        dataset_fingerprint=spec.dataset.fingerprint,
        in_sample=in_sample,
        out_of_sample=out_of_sample,
        walk_forward=walk_forward,
        monte_carlo=monte_carlo,
        parameter_stability=stability,
        segmentation=segmentation,
        cost_stress=stress,
        gates=gates,
        verdict=verdict,
        limitations=tuple(limitations),
    )
