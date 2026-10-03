"""Deterministic XAU Strategy v2 feature contract.

This module does not replace market_context or cognition.  It freezes a
backtestable strategy hypothesis on top of those observations so every added
layer can be ablated and validated statistically.

Important evidence rule:
- EMA ladder, HTF slope, SMC structure, tick-volume flow and intraday price
  action are exposed separately.
- Volume Profile and liquidity levels are context/geometry by default, not
  directional votes.
- Spot XAU tick volume and GC futures participation are never described as
  centralized global spot order flow.
- No live execution is available here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from src.modules.strategy.xau_intraday import XAUIntradayAssessment


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _direction(value: Any) -> float:
    text = str(value or "").lower()
    if text in {"bullish", "up", "inflow", "long"}:
        return 1.0
    if text in {"bearish", "down", "outflow", "short"}:
        return -1.0
    return 0.0


@dataclass(frozen=True)
class XAUFeatureFlags:
    intraday_structure: bool = True
    htf_momentum: bool = True
    ema_ladder: bool = True
    smart_money_structure: bool = True
    flow_proxy: bool = True
    volume_profile_context: bool = True
    liquidity_context: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "intraday_structure": self.intraday_structure,
            "htf_momentum": self.htf_momentum,
            "ema_ladder": self.ema_ladder,
            "smart_money_structure": self.smart_money_structure,
            "flow_proxy": self.flow_proxy,
            "volume_profile_context": self.volume_profile_context,
            "liquidity_context": self.liquidity_context,
        }


@dataclass(frozen=True)
class XAUStrategyV2Spec:
    """Research-only, engine-neutral strategy hypothesis."""

    strategy_id: str = "panwatch-xau-v2"
    schema_version: str = "2.0"
    features: XAUFeatureFlags = field(default_factory=XAUFeatureFlags)
    intraday_weight: float = 0.30
    htf_momentum_weight: float = 0.20
    ema_ladder_weight: float = 0.20
    smart_money_weight: float = 0.20
    flow_weight: float = 0.10
    score_threshold: float = 0.30
    min_directional_features: int = 3
    htf_conflict_threshold: float = 0.55
    live: bool = field(default=False, init=False)
    broker_orders_enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        weights = (
            self.intraday_weight,
            self.htf_momentum_weight,
            self.ema_ladder_weight,
            self.smart_money_weight,
            self.flow_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("strategy weights must be non-negative with positive total")
        if not 0 < self.score_threshold <= 1:
            raise ValueError("score_threshold must be within (0,1]")
        if self.min_directional_features < 1:
            raise ValueError("min_directional_features must be >= 1")
        if not 0 <= self.htf_conflict_threshold <= 1:
            raise ValueError("htf_conflict_threshold must be within [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "schema_version": self.schema_version,
            "features": self.features.to_dict(),
            "weights": {
                "intraday_structure": self.intraday_weight,
                "htf_momentum": self.htf_momentum_weight,
                "ema_ladder": self.ema_ladder_weight,
                "smart_money_structure": self.smart_money_weight,
                "flow_proxy": self.flow_weight,
            },
            "decision": {
                "score_threshold": self.score_threshold,
                "min_directional_features": self.min_directional_features,
                "htf_conflict_threshold": self.htf_conflict_threshold,
            },
            "context_only": ["volume_profile_context", "liquidity_context"],
            "execution": {
                "live": False,
                "broker_orders_enabled": False,
                "simulation_allowed": True,
            },
            "evidence_policy": {
                "volume_profile_directional_vote": False,
                "liquidity_levels_directional_vote": False,
                "spot_tick_volume_is_global_order_flow": False,
                "gc_futures_is_xauusd_execution_quote": False,
                "same_input_library_votes_are_independent": False,
            },
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class XAUFeatureObservation:
    name: str
    available: bool
    score: float | None
    contributing: bool
    provenance_group: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class XAUStrategyV2Assessment:
    status: str
    candidate: str
    score: float | None
    directional_feature_count: int
    observations: dict[str, XAUFeatureObservation]
    block_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    strategy_fingerprint: str
    context: dict[str, Any]
    live_execution_allowed: bool = False


def _intraday_observation(
    assessment: XAUIntradayAssessment | None,
) -> XAUFeatureObservation:
    if assessment is None:
        return XAUFeatureObservation(
            "intraday_structure", False, None, True, "intraday_price",
            {"reason": "assessment_missing"},
        )
    frames = assessment.frame_states
    required = ("1m", "5m", "15m")
    if not all(name in frames for name in required):
        return XAUFeatureObservation(
            "intraday_structure", False, None, True, "intraday_price",
            {"reason": "required_frames_missing"},
        )

    weights = {"1m": 0.20, "5m": 0.35, "15m": 0.45}
    score = sum(weights[name] * _direction(frames[name].direction) for name in required)
    return XAUFeatureObservation(
        "intraday_structure",
        True,
        round(_clip(score), 4),
        True,
        "intraday_price",
        {
            "directions": {name: frames[name].direction for name in required},
            "baseline_candidate": assessment.candidate,
        },
    )


def _htf_states(market_context: dict[str, Any]) -> list[tuple[str, dict[str, Any], float]]:
    bias = dict(market_context.get("bias") or {})
    return [
        ("monthly", dict(bias.get("monthly") or {}), 0.34),
        ("weekly", dict(bias.get("weekly") or {}), 0.30),
        ("daily", dict(bias.get("daily") or {}), 0.24),
        ("h4", dict(bias.get("h4") or {}), 0.08),
        ("h1", dict(bias.get("h1") or {}), 0.04),
    ]


def _htf_momentum_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    numerator = 0.0
    denominator = 0.0
    detail: dict[str, Any] = {}
    for name, state, weight in _htf_states(market_context):
        if not state.get("available"):
            continue
        slope = float(state.get("slope_20") or 0.0)
        score = _clip(slope / 0.04)
        numerator += weight * score
        denominator += weight
        detail[name] = {"slope_20": slope, "score": round(score, 4)}

    if denominator == 0:
        return XAUFeatureObservation(
            "htf_momentum", False, None, True, "higher_timeframe_price",
            {"reason": "htf_slope_unavailable"},
        )
    return XAUFeatureObservation(
        "htf_momentum",
        True,
        round(_clip(numerator / denominator), 4),
        True,
        "higher_timeframe_price",
        detail,
    )


def _ladder_score(state: dict[str, Any]) -> float | None:
    ema = dict(state.get("ema") or {})
    close = state.get("close")
    pairs = (("9", "21"), ("21", "50"), ("50", "200"), ("200", "1000"))
    votes: list[float] = []
    for fast, slow in pairs:
        a = ema.get(fast)
        b = ema.get(slow)
        if a is None or b is None:
            continue
        votes.append(1.0 if float(a) > float(b) else -1.0 if float(a) < float(b) else 0.0)
    if close is not None:
        for period in ("50", "200", "1000"):
            value = ema.get(period)
            if value is not None:
                votes.append(
                    1.0 if float(close) > float(value)
                    else -1.0 if float(close) < float(value)
                    else 0.0
                )
    if not votes:
        return None
    return sum(votes) / len(votes)


def _ema_ladder_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    numerator = 0.0
    denominator = 0.0
    detail: dict[str, Any] = {}
    for name, state, weight in _htf_states(market_context):
        if not state.get("available"):
            continue
        score = _ladder_score(state)
        if score is None:
            continue
        numerator += weight * score
        denominator += weight
        detail[name] = {
            "score": round(score, 4),
            "ema": dict(state.get("ema") or {}),
            "close": state.get("close"),
        }
    if denominator == 0:
        return XAUFeatureObservation(
            "ema_ladder", False, None, True, "higher_timeframe_ema",
            {"reason": "ema_ladder_unavailable"},
        )
    return XAUFeatureObservation(
        "ema_ladder",
        True,
        round(_clip(numerator / denominator), 4),
        True,
        "higher_timeframe_ema",
        detail,
    )


def _smart_money_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    smc = dict(market_context.get("smart_money") or {})
    if not smc.get("available"):
        return XAUFeatureObservation(
            "smart_money_structure", False, None, True, "derived_price_structure",
            {"reason": "smart_money_structure_unavailable"},
        )

    score = 0.0
    bos = str(smc.get("break_of_structure") or "none")
    sweep = str(smc.get("liquidity_sweep") or "none")
    displacement = str(smc.get("displacement") or "none")
    score += 0.40 * _direction(bos)
    if sweep == "sell_side_sweep":
        score += 0.30
    elif sweep == "buy_side_sweep":
        score -= 0.30
    score += 0.30 * _direction(displacement)

    return XAUFeatureObservation(
        "smart_money_structure",
        True,
        round(_clip(score), 4),
        True,
        "derived_price_structure",
        {
            "break_of_structure": bos,
            "liquidity_sweep": sweep,
            "displacement": displacement,
            "dealing_range": smc.get("dealing_range"),
            "fair_value_gaps": smc.get("fair_value_gaps", []),
            "note": "score excludes market_context's flow/profile additions to avoid double counting",
        },
    )


def _flow_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    flow = dict(market_context.get("cash_flow") or {})
    if not flow.get("available"):
        return XAUFeatureObservation(
            "flow_proxy", False, None, True, "volume_activity_proxy",
            {"reason": "flow_proxy_unavailable"},
        )
    return XAUFeatureObservation(
        "flow_proxy",
        True,
        round(_clip(float(flow.get("score") or 0.0)), 4),
        True,
        "volume_activity_proxy",
        {
            "direction": flow.get("direction"),
            "agreement": flow.get("agreement"),
            "spot_tick": flow.get("spot_tick"),
            "gc_futures": flow.get("gc_futures"),
            "centralized_global_spot_order_flow": False,
        },
    )


def _profile_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    profile = dict(market_context.get("volume_profile") or {})
    if not profile.get("available"):
        return XAUFeatureObservation(
            "volume_profile_context", False, None, False, "volume_activity_proxy",
            {"reason": profile.get("reason", "profile_unavailable")},
        )
    return XAUFeatureObservation(
        "volume_profile_context",
        True,
        None,
        False,
        "volume_activity_proxy",
        {
            "poc": profile.get("poc"),
            "vah": profile.get("vah"),
            "val": profile.get("val"),
            "location": profile.get("location"),
            "hvn": profile.get("hvn", []),
            "lvn": profile.get("lvn", []),
            "directional_vote": False,
        },
    )


def _liquidity_observation(market_context: dict[str, Any]) -> XAUFeatureObservation:
    liquidity = dict(market_context.get("liquidity") or {})
    if not liquidity.get("available"):
        return XAUFeatureObservation(
            "liquidity_context", False, None, False, "price_geometry",
            {"reason": "liquidity_map_unavailable"},
        )
    return XAUFeatureObservation(
        "liquidity_context",
        True,
        None,
        False,
        "price_geometry",
        {
            "levels": liquidity.get("levels", []),
            "equal_highs": liquidity.get("equal_highs", []),
            "equal_lows": liquidity.get("equal_lows", []),
            "directional_vote": False,
        },
    )


def assess_xau_strategy_v2(
    *,
    spec: XAUStrategyV2Spec,
    intraday: XAUIntradayAssessment | None,
    market_context: dict[str, Any] | None,
) -> XAUStrategyV2Assessment:
    context = dict(market_context or {})
    flags = spec.features

    observations = {
        "intraday_structure": _intraday_observation(intraday),
        "htf_momentum": _htf_momentum_observation(context),
        "ema_ladder": _ema_ladder_observation(context),
        "smart_money_structure": _smart_money_observation(context),
        "flow_proxy": _flow_observation(context),
        "volume_profile_context": _profile_observation(context),
        "liquidity_context": _liquidity_observation(context),
    }

    enabled = {
        "intraday_structure": flags.intraday_structure,
        "htf_momentum": flags.htf_momentum,
        "ema_ladder": flags.ema_ladder,
        "smart_money_structure": flags.smart_money_structure,
        "flow_proxy": flags.flow_proxy,
        "volume_profile_context": flags.volume_profile_context,
        "liquidity_context": flags.liquidity_context,
    }
    weights = {
        "intraday_structure": spec.intraday_weight,
        "htf_momentum": spec.htf_momentum_weight,
        "ema_ladder": spec.ema_ladder_weight,
        "smart_money_structure": spec.smart_money_weight,
        "flow_proxy": spec.flow_weight,
    }

    warnings: list[str] = []
    block_reasons: list[str] = []
    numerator = 0.0
    denominator = 0.0
    directional_count = 0

    for name, weight in weights.items():
        if not enabled[name]:
            continue
        obs = observations[name]
        if not obs.available or obs.score is None:
            warnings.append(f"{name}_unavailable")
            continue
        directional_count += 1
        numerator += weight * obs.score
        denominator += weight

    score = None if denominator == 0 else _clip(numerator / denominator)

    if directional_count < spec.min_directional_features:
        block_reasons.append("insufficient_directional_features")

    htf_guard_values = [
        observations["htf_momentum"].score if flags.htf_momentum else None,
        observations["ema_ladder"].score if flags.ema_ladder else None,
    ]
    htf_values = [value for value in htf_guard_values if value is not None]
    htf_guard = sum(htf_values) / len(htf_values) if htf_values else None

    candidate = "none"
    if score is not None and directional_count >= spec.min_directional_features:
        if score >= spec.score_threshold:
            candidate = "long_setup"
        elif score <= -spec.score_threshold:
            candidate = "short_setup"

    if (
        candidate == "long_setup"
        and htf_guard is not None
        and htf_guard <= -spec.htf_conflict_threshold
    ):
        block_reasons.append("strong_htf_conflict_long")
    elif (
        candidate == "short_setup"
        and htf_guard is not None
        and htf_guard >= spec.htf_conflict_threshold
    ):
        block_reasons.append("strong_htf_conflict_short")

    if intraday is not None and intraday.blocked:
        block_reasons.extend(intraday.block_reasons)

    if block_reasons:
        candidate = "none"

    selected_context = {
        "volume_profile": (
            observations["volume_profile_context"].detail
            if flags.volume_profile_context
            else {"enabled": False}
        ),
        "liquidity": (
            observations["liquidity_context"].detail
            if flags.liquidity_context
            else {"enabled": False}
        ),
        "volume_note": context.get("volume_note"),
    }

    return XAUStrategyV2Assessment(
        status="blocked" if block_reasons else "ready",
        candidate=candidate,
        score=round(score, 4) if score is not None else None,
        directional_feature_count=directional_count,
        observations=observations,
        block_reasons=tuple(dict.fromkeys(block_reasons)),
        warnings=tuple(dict.fromkeys(warnings)),
        strategy_fingerprint=spec.fingerprint,
        context=selected_context,
        live_execution_allowed=False,
    )


def xau_v2_ablation_specs() -> tuple[tuple[str, XAUStrategyV2Spec], ...]:
    """Cumulative variants for edge attribution.

    The order is intentional: do not jump directly to the full model when
    researching whether each extra layer contributes out-of-sample value.
    """

    return (
        (
            "baseline_intraday",
            XAUStrategyV2Spec(
                features=XAUFeatureFlags(
                    intraday_structure=True,
                    htf_momentum=False,
                    ema_ladder=False,
                    smart_money_structure=False,
                    flow_proxy=False,
                    volume_profile_context=False,
                    liquidity_context=False,
                ),
                min_directional_features=1,
            ),
        ),
        (
            "plus_htf_momentum",
            XAUStrategyV2Spec(
                features=XAUFeatureFlags(
                    intraday_structure=True,
                    htf_momentum=True,
                    ema_ladder=False,
                    smart_money_structure=False,
                    flow_proxy=False,
                    volume_profile_context=False,
                    liquidity_context=False,
                ),
                min_directional_features=2,
            ),
        ),
        (
            "plus_ema_ladder",
            XAUStrategyV2Spec(
                features=XAUFeatureFlags(
                    intraday_structure=True,
                    htf_momentum=True,
                    ema_ladder=True,
                    smart_money_structure=False,
                    flow_proxy=False,
                    volume_profile_context=False,
                    liquidity_context=False,
                ),
                min_directional_features=3,
            ),
        ),
        (
            "plus_smart_money",
            XAUStrategyV2Spec(
                features=XAUFeatureFlags(
                    intraday_structure=True,
                    htf_momentum=True,
                    ema_ladder=True,
                    smart_money_structure=True,
                    flow_proxy=False,
                    volume_profile_context=False,
                    liquidity_context=True,
                ),
                min_directional_features=3,
            ),
        ),
        (
            "plus_flow_and_profile_context",
            XAUStrategyV2Spec(),
        ),
    )
