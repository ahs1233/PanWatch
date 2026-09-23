"""GEN1 gold evidence fusion.

This layer combines *independent evidence families* rather than counting every
indicator as an independent vote. XAUT trade flow, footprint, volume profile
and raw book all belong to one centralized XAUT microstructure family.

Scores are research heuristics in [-1, 1]. Confidence is an inspectable
decision score, NOT a validated win probability.
"""
from __future__ import annotations

from typing import Any


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _empirical_cognitive_calibration(
    raw_confidence: float,
    memory: dict[str, Any],
) -> tuple[float, dict[str, Any]]:
    """Conservatively shrink cognitive confidence toward observed paper outcomes.

    The historical bins were generated from the cognition confidence itself,
    so calibration is applied to that component before evidence-strength
    blending. The result remains a decision score, not a win probability.
    """
    raw = _clip(raw_confidence, 0.05, 0.95)
    bins = list(memory.get("calibration_bins") or [])
    sample_count = int(_num(memory.get("calibration_sample_count"), 0))
    if sample_count < 20 or not bins:
        return raw, {
            "applied": False,
            "reason": "insufficient_calibration_history",
            "sample_count": sample_count,
            "raw_cognitive_confidence": round(raw, 4),
            "calibrated_cognitive_confidence": round(raw, 4),
        }

    selected = None
    for row in bins:
        lower = _num(row.get("lower"), 0.0)
        upper = _num(row.get("upper"), 1.0)
        if lower <= raw < upper or (raw >= 1.0 and upper >= 1.0):
            selected = row
            break
    if not selected:
        return raw, {
            "applied": False,
            "reason": "matching_calibration_bin_unavailable",
            "sample_count": sample_count,
            "raw_cognitive_confidence": round(raw, 4),
            "calibrated_cognitive_confidence": round(raw, 4),
        }

    bin_count = int(_num(selected.get("count"), 0))
    observed = selected.get("observed_rate")
    if bin_count < 8 or observed is None:
        return raw, {
            "applied": False,
            "reason": "matching_bin_too_small",
            "sample_count": sample_count,
            "bin_count": bin_count,
            "raw_cognitive_confidence": round(raw, 4),
            "calibrated_cognitive_confidence": round(raw, 4),
        }

    observed_rate = _clip(_num(observed, raw), 0.05, 0.95)
    ece = _clip(_num(memory.get("expected_calibration_error"), 0.25), 0.0, 1.0)
    history_weight = min(0.45, (bin_count / (bin_count + 30.0)) * 0.60)
    quality_weight = max(0.25, 1.0 - ece)
    alpha = history_weight * quality_weight
    calibrated = _clip((1.0 - alpha) * raw + alpha * observed_rate, 0.05, 0.95)
    return calibrated, {
        "applied": True,
        "method": "paper_cognition_bin_shrinkage_v1",
        "sample_count": sample_count,
        "bin_count": bin_count,
        "bin_index": selected.get("index"),
        "observed_rate": round(observed_rate, 4),
        "alpha": round(alpha, 4),
        "raw_cognitive_confidence": round(raw, 4),
        "calibrated_cognitive_confidence": round(calibrated, 4),
        "not_a_win_probability": True,
    }


def _candidate_sign(candidate: str) -> float:
    if candidate == "long_setup":
        return 1.0
    if candidate == "short_setup":
        return -1.0
    return 0.0


def _xaut_score(technical: dict[str, Any]) -> dict[str, Any]:
    xaut = technical.get("xaut_order_flow") or {}
    if not xaut or xaut.get("status") != "ready":
        return {
            "available": False,
            "score": 0.0,
            "reliability": 0.0,
            "components": {},
            "reason": "xaut_unavailable",
        }

    components: dict[str, float] = {}
    weights: dict[str, float] = {}

    flow5 = (xaut.get("flow") or {}).get("5m") or {}
    if flow5.get("available"):
        components["executed_flow_5m"] = _clip(_num(flow5.get("delta_ratio")))
        weights["executed_flow_5m"] = 0.42

    footprint = xaut.get("footprint") or {}
    levels = footprint.get("levels") or []
    if footprint.get("available") and levels:
        total = sum(max(0.0, _num(row.get("total_volume"))) for row in levels)
        delta = sum(_num(row.get("delta")) for row in levels)
        if total > 0:
            components["footprint_delta"] = _clip(delta / total)
            weights["footprint_delta"] = 0.18

    profile = xaut.get("volume_profile") or {}
    profile_levels = profile.get("levels") or []
    if profile.get("status") == "ready" and profile_levels:
        volume = sum(max(0.0, _num(row.get("volume"))) for row in profile_levels)
        delta = sum(_num(row.get("delta")) for row in profile_levels)
        if volume > 0:
            components["profile_delta"] = _clip(delta / volume)
            weights["profile_delta"] = 0.10

    book10 = (xaut.get("raw_book") or {}).get("pm10") or {}
    if "imbalance" in book10:
        components["raw_book_pm10"] = _clip(_num(book10.get("imbalance")))
        weights["raw_book_pm10"] = 0.22

    absorption = str((xaut.get("absorption") or {}).get("state") or "none")
    if absorption == "possible_sell_absorption":
        components["absorption"] = 0.35
        weights["absorption"] = 0.08
    elif absorption == "possible_buy_absorption":
        components["absorption"] = -0.35
        weights["absorption"] = 0.08

    if not weights:
        return {
            "available": False,
            "score": 0.0,
            "reliability": 0.0,
            "components": components,
            "reason": "no_usable_xaut_components",
        }

    denom = sum(weights.values()) or 1.0
    raw = sum(components[name] * weights[name] for name in weights) / denom

    tape = xaut.get("trade_tape") or {}
    trade_count = max(0, int(_num(tape.get("trade_count"), 0)))
    coverage = max(0.0, _num(tape.get("coverage_seconds"), 0.0))
    sample_reliability = min(1.0, trade_count / 250.0)
    coverage_reliability = min(1.0, coverage / 900.0)
    reliability = 0.45 + 0.25 * sample_reliability + 0.20 * coverage_reliability

    basis_bps = abs(_num((xaut.get("basis") or {}).get("basis_bps"), 0.0))
    if basis_bps > 20:
        reliability *= 0.55
    elif basis_bps > 10:
        reliability *= 0.75

    # One centralized proxy venue must never receive full global-flow weight.
    reliability = _clip(reliability, 0.20, 0.82)
    score = _clip(raw * reliability)

    return {
        "available": True,
        "score": round(score, 4),
        "raw_score": round(raw, 4),
        "reliability": round(reliability, 4),
        "components": {k: round(v, 4) for k, v in components.items()},
        "source_family": "bitfinex_xaut_centralized_microstructure",
        "global_xauusd_order_flow": False,
        "basis_bps": round(basis_bps, 4),
    }


def build_gen1_evidence_fusion(
    technical: dict[str, Any],
    macro: dict[str, Any],
    fusion: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
    require_xaut: bool = True,
) -> dict[str, Any]:
    """Fuse independent families and produce a conservative GEN1 decision."""
    memory = memory or {}
    cognition = fusion.get("cognition") or {}
    market_context = technical.get("market_context") or {}
    bias = market_context.get("bias") or {}
    smart = market_context.get("smart_money") or {}
    edge = cognition.get("directional_edge") or {}

    candidate = str(fusion.get("technical_candidate") or technical.get("candidate") or "none")
    candidate_sign = _candidate_sign(candidate)

    families: list[dict[str, Any]] = []

    def add(name: str, score: float, weight: float, *, available: bool, source_family: str, detail: dict | None = None):
        families.append({
            "name": name,
            "score": round(_clip(score), 4),
            "weight": round(float(weight), 4),
            "available": bool(available),
            "source_family": source_family,
            "detail": detail or {},
        })

    htf_available = bool(bias) and any((bias.get(k) or {}).get("available") for k in ("monthly", "weekly", "daily", "h4", "h1"))
    add(
        "higher_timeframe_structure",
        _num(bias.get("today_score"), _num(bias.get("composite_score"), 0.0)),
        0.25,
        available=htf_available,
        source_family="xau_spot_htf_ohlc",
        detail={"direction": bias.get("today_direction") or bias.get("composite_direction")},
    )

    edge_score = _num(edge.get("score"), 0.0)
    add(
        "intraday_cognition",
        edge_score,
        0.22,
        available=bool(cognition),
        source_family="xau_intraday_ohlc_micro",
        detail={"strength": edge.get("strength"), "direction": edge.get("direction")},
    )

    smart_score = _num(smart.get("manual_score"), _num(smart.get("score"), 0.0))
    add(
        "smc_liquidity",
        smart_score,
        0.14,
        available=bool(smart),
        source_family="xau_structure_liquidity",
        detail={"bias": smart.get("bias"), "validated_liquidity_sweep": smart.get("validated_liquidity_sweep")},
    )

    macro_ready = bool(fusion.get("macro_ready"))
    macro_score = _clip(_num(macro.get("bias"), 0.0)) * _clip(_num(macro.get("confidence"), 0.0), 0.0, 1.0)
    add(
        "external_macro",
        macro_score,
        0.12,
        available=macro_ready,
        source_family="ahmed_toolbox_external_macro",
        detail={"bias_label": macro.get("bias_label"), "confidence": macro.get("confidence")},
    )

    xaut = _xaut_score(technical)
    add(
        "xaut_microstructure",
        _num(xaut.get("score"), 0.0),
        0.19,
        available=bool(xaut.get("available")),
        source_family="bitfinex_xaut_centralized_microstructure",
        detail=xaut,
    )

    memory_samples = max(
        int(_num(memory.get("similar_samples"), 0)),
        int(_num(memory.get("calibration_sample_count"), 0)),
        int(_num(memory.get("trade_count"), 0)),
    )
    posterior = memory.get("posterior_win_probability")
    memory_available = bool(memory.get("available", True)) and posterior is not None and memory_samples >= 5 and candidate_sign != 0
    memory_edge = (_num(posterior, 0.5) - 0.5) * 2.0 if memory_available else 0.0
    add(
        "empirical_memory",
        candidate_sign * memory_edge,
        0.08,
        available=memory_available,
        source_family="panwatch_paper_replay_memory",
        detail={
            "samples": memory_samples,
            "posterior_win_probability": posterior,
            "brier_score": memory.get("brier_score"),
            "expected_calibration_error": memory.get("expected_calibration_error"),
        },
    )

    available = [item for item in families if item["available"]]
    weight_sum = sum(item["weight"] for item in available)
    score = (
        sum(item["score"] * item["weight"] for item in available) / weight_sum
        if weight_sum > 0 else 0.0
    )
    score = _clip(score)
    coverage = weight_sum / sum(item["weight"] for item in families)

    directional = [item for item in available if abs(item["score"]) >= 0.10]
    if candidate_sign:
        agreeing = [item for item in directional if item["score"] * candidate_sign > 0]
        opposing = [item for item in directional if item["score"] * candidate_sign < 0]
        agreement_ratio = len(agreeing) / len(directional) if directional else 0.5
    else:
        agreeing, opposing = [], []
        agreement_ratio = 0.5

    conflicts = [
        {
            "family": item["name"],
            "score": item["score"],
            "source_family": item["source_family"],
        }
        for item in opposing
        if abs(item["score"]) >= 0.18
    ]

    evidence_direction = "bullish" if score >= 0.10 else "bearish" if score <= -0.10 else "neutral"
    cognitive_confidence = _clip(_num(fusion.get("cognitive_confidence"), 0.5), 0.05, 0.95)
    calibrated_cognitive, calibration = _empirical_cognitive_calibration(
        cognitive_confidence,
        memory,
    )
    memory_probability = _clip(_num(posterior, 0.5), 0.05, 0.95) if posterior is not None else 0.5
    evidence_strength = 0.5 + 0.5 * abs(score)
    decision_confidence = (
        0.64 * calibrated_cognitive
        + 0.26 * evidence_strength
        + 0.10 * memory_probability
    )
    decision_confidence *= 0.75 + 0.25 * coverage
    if conflicts:
        decision_confidence *= max(0.65, 1.0 - 0.08 * len(conflicts))
    decision_confidence = _clip(decision_confidence, 0.05, 0.95)

    decision = "WAIT"
    reasons: list[str] = []
    if candidate_sign == 0:
        reasons.append("no_directional_candidate")
    if not bool(fusion.get("paper_entry_allowed")):
        reasons.append("gen1_meta_controller_not_eligible")
    if require_xaut and not xaut.get("available"):
        reasons.append("xaut_microstructure_required_but_unavailable")

    candidate_evidence = score * candidate_sign if candidate_sign else 0.0
    strong_opposition = any(
        item["name"] in {"higher_timeframe_structure", "xaut_microstructure"}
        and item["score"] * candidate_sign <= -0.25
        for item in available
    ) if candidate_sign else False
    if strong_opposition:
        reasons.append("strong_independent_evidence_conflict")

    if (
        candidate_sign != 0
        and bool(fusion.get("paper_entry_allowed"))
        and (not require_xaut or bool(xaut.get("available")))
        and candidate_evidence >= 0.10
        and not strong_opposition
    ):
        decision = "LONG" if candidate_sign > 0 else "SHORT"
        reasons.append("independent_evidence_confirms_candidate")
    elif candidate_sign != 0 and candidate_evidence < 0.10:
        reasons.append("evidence_confirmation_below_threshold")

    return {
        "version": "gen1-evidence-fusion-v1",
        "decision": decision,
        "candidate": candidate,
        "score": round(score, 4),
        "direction": evidence_direction,
        "coverage": round(coverage, 4),
        "agreement_ratio": round(agreement_ratio, 4),
        "decision_confidence": round(decision_confidence, 4),
        "is_validated_win_probability": False,
        "confidence_kind": (
            "heuristic_score_with_empirical_cognitive_bin_shrinkage"
            if calibration.get("applied")
            else "heuristic_unvalidated_score"
        ),
        "calibration": calibration,
        "families": families,
        "conflicts": conflicts,
        "reasons": list(dict.fromkeys(reasons)),
        "xaut": xaut,
        "require_xaut": bool(require_xaut),
        "execution_allowed": False,
    }
