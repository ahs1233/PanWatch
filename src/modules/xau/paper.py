"""Weekly XAU/USD paper-trading league engine.

The engine is intentionally isolated from live execution. It consumes the same
research state as the terminal, simulates conservative bid/ask fills, and keeps
execution_allowed=False throughout.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy.exc import IntegrityError
from src.modules.xau.cognition import (
    build_market_state_vector,
    state_vector_similarity,
)
from src.modules.xau.service import (
    build_decision_fusion,
    get_macro_context,
    get_xau_snapshot,
)
from src.modules.xau.paper_store import (
    open_xau_paper_session,
    open_xau_replay_session,
    paper_store_is_external,
    replay_store_is_external,
)
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
    XAUReplayEpisode,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

PAPER_ENGINE_VERSION = "0.9.0"


def _utc_naive(now: datetime | None = None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _local_now(settings: Settings, now: datetime | None = None) -> datetime:
    tz = ZoneInfo(settings.xau_paper_timezone)
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(tz)


def _week_key(settings: Settings, now: datetime | None = None) -> str:
    local = _local_now(settings, now)
    iso = local.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _week_start_utc_naive(settings: Settings, now: datetime | None = None) -> datetime:
    local = _local_now(settings, now)
    monday = (local - timedelta(days=local.weekday())).replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    return monday.astimezone(timezone.utc).replace(tzinfo=None)


def _number(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


_TRANSIENT_SIGNAL_REJECTIONS = frozenset({
    "bid_ask_unavailable",
    "indicative_spot_stale",
    "market_closed_or_rollover",
    "stale_bid_ask",
    "spread_too_wide",
})


def _can_revalidate_signal(
    existing_accepted: bool,
    existing_reason: str,
    accepted_now: bool,
) -> bool:
    return (
        not existing_accepted
        and accepted_now
        and existing_reason in _TRANSIENT_SIGNAL_REJECTIONS
    )


def _entry_gate_reason(
    *,
    candidate: str,
    fusion_state: str,
    spot: dict,
    has_open_position: bool,
    max_spread_bps: float,
) -> tuple[bool, str]:
    if has_open_position:
        return False, "position_already_open"
    if candidate not in {"long_setup", "short_setup"}:
        return False, fusion_state or "no_setup"

    provider_health = spot.get("provider_health") or []
    for row in provider_health:
        if not isinstance(row, dict) or not bool(row.get("has_bid_ask")):
            continue
        market_state = str(row.get("market_state") or "").lower()
        if market_state in {"closed", "market_closed", "maintenance", "rollover"}:
            return False, "market_closed_or_rollover"

    fill_state = str(spot.get("fill_state") or "")
    if fill_state == "market_closed_or_rollover":
        return False, "market_closed_or_rollover"
    if fill_state == "stale_bid_ask":
        return False, "stale_bid_ask"
    if bool(spot.get("is_stale")):
        return False, "indicative_spot_stale"
    if _number(spot.get("bid")) is None or _number(spot.get("ask")) is None:
        return False, "bid_ask_unavailable"
    spread_bps = _spot_spread_bps(spot)
    if spread_bps is not None and spread_bps > float(max_spread_bps):
        return False, "spread_too_wide"
    if fusion_state not in {"setup_macro_support", "setup_macro_neutral"}:
        return False, fusion_state or "fusion_not_eligible"
    return True, ""


def _paper_entry_price(side: str, spot: dict) -> float | None:
    """Paper entries require an actual indicative bid/ask side.

    Mid-only references are useful for context, but are not accepted as fills.
    """
    if side == "long":
        return _number(spot.get("ask"))
    return _number(spot.get("bid"))


def _paper_mark_price(side: str, spot: dict) -> float | None:
    if side == "long":
        return _number(spot.get("bid")) or _number(spot.get("price"))
    return _number(spot.get("ask")) or _number(spot.get("price"))


def _paper_context_mark_price(
    side: str,
    spot: dict,
    analysis_reference: dict | None = None,
) -> tuple[float | None, str]:
    """Best available non-executable mark for equity/PnL display.

    Fresh bid/ask is preferred. A fresh analytical mid may mark an open paper
    position, but it can never trigger or price a simulated exit.
    """
    spot = spot or {}
    if spot and not bool(spot.get("is_stale")):
        mark = _paper_mark_price(side, spot)
        if mark is not None:
            return mark, "fill_quote"

    reference = analysis_reference or {}
    if (
        reference
        and not bool(reference.get("is_stale"))
        and _number(reference.get("price")) is not None
    ):
        return _number(reference.get("price")), "analysis_reference"

    return None, "unavailable"


def _paper_exit_quote(side: str, spot: dict) -> float | None:
    """Executable-side indicative quote used for simulated exits."""
    if side == "long":
        return _number(spot.get("bid"))
    return _number(spot.get("ask"))


def _paper_management_quote(side: str, spot: dict | None) -> float | None:
    """Fresh paper quote allowed to drive stops, targets, guardian and time exits."""
    spot = spot or {}
    if not spot or bool(spot.get("is_stale")):
        return None
    fill_state = str(spot.get("fill_state") or "")
    if fill_state and fill_state != "ready":
        return None
    return _paper_exit_quote(side, spot)


def _weekly_reset_fill_price(side: str, spot: dict | None) -> float | None:
    """Weekly rollover requires the same fresh paper-fill contract as management."""
    return _paper_management_quote(side, spot)


def _position_age_minutes(opened_at: datetime, now_utc: datetime) -> float:
    opened = opened_at
    if opened.tzinfo is not None:
        opened = opened.astimezone(timezone.utc).replace(tzinfo=None)
    now_value = now_utc
    if now_value.tzinfo is not None:
        now_value = now_value.astimezone(timezone.utc).replace(tzinfo=None)
    return max(0.0, (now_value - opened).total_seconds() / 60.0)


def _spot_spread_bps(spot: dict) -> float | None:
    direct = _number(spot.get("spread_bps"))
    if direct is not None:
        return direct
    bid = _number(spot.get("bid"))
    ask = _number(spot.get("ask"))
    mid = _number(spot.get("price"))
    if bid is None or ask is None:
        return None
    basis = mid or ((bid + ask) / 2.0)
    if basis <= 0:
        return None
    return max(0.0, (ask - bid) / basis * 10_000.0)


def _pnl(side: str, entry: float, mark: float, quantity: float) -> float:
    if side == "long":
        return (mark - entry) * quantity
    return (entry - mark) * quantity


def _position_guardian(
    position,
    fusion: dict,
    exit_quote: float,
) -> dict:
    """Paper-only position management using fresh executable-side quotes.

    The guardian never manufactures fills. It can tighten a protective stop
    only when current bid/ask implies sufficient open profit, and it can request
    an early exit only when the current decision stack independently qualifies
    an opposite setup.
    """
    side = str(getattr(position, "side", "") or "")
    entry = _number(getattr(position, "entry_price", None))
    quantity = _number(getattr(position, "quantity_oz", None))
    risk_usd = _number(getattr(position, "risk_usd", None))
    current_stop = _number(getattr(position, "stop_loss", None))
    target = _number(getattr(position, "target_price", None))
    quote = _number(exit_quote)

    result = {
        "action": "hold",
        "reason": "no_management_trigger",
        "current_r": None,
        "new_stop_loss": None,
        "exit_requested": False,
        "exit_reason": None,
    }
    if (
        side not in {"long", "short"}
        or entry is None
        or quantity is None
        or quantity <= 0
        or risk_usd is None
        or risk_usd <= 0
        or current_stop is None
        or quote is None
    ):
        result["reason"] = "insufficient_position_or_quote_data"
        return result

    pnl = _pnl(side, entry, quote, quantity)
    current_r = pnl / risk_usd
    result["current_r"] = round(current_r, 4)

    initial_risk_distance = risk_usd / quantity
    protective_stop = current_stop
    stop_reason = None

    if current_r >= 1.5:
        candidate_stop = (
            entry + 0.5 * initial_risk_distance
            if side == "long"
            else entry - 0.5 * initial_risk_distance
        )
        if side == "long":
            improved = max(current_stop, candidate_stop)
            if target is None or improved < target:
                protective_stop = improved
        else:
            improved = min(current_stop, candidate_stop)
            if target is None or improved > target:
                protective_stop = improved
        if protective_stop != current_stop:
            stop_reason = "lock_half_r_after_1_5r"
    elif current_r >= 1.0:
        candidate_stop = entry
        if side == "long":
            protective_stop = max(current_stop, candidate_stop)
        else:
            protective_stop = min(current_stop, candidate_stop)
        if protective_stop != current_stop:
            stop_reason = "breakeven_after_1r"

    candidate = str(fusion.get("technical_candidate") or "none")
    opposite_candidate = "short_setup" if side == "long" else "long_setup"
    meta_decision = str(fusion.get("meta_decision") or "observe")
    confidence = _number(fusion.get("cognitive_confidence"), 0.0) or 0.0
    cognition = fusion.get("cognition") or {}
    meta = cognition.get("meta_controller") or {}
    data_quality = _number(
        (cognition.get("data_quality") or {}).get("score"),
        0.0,
    ) or 0.0
    entry_threshold = _number(meta.get("min_confidence"), 0.58) or 0.58
    # Reversal exits require more evidence than opening a fresh position.
    # This hysteresis reduces flip-flop around noisy threshold crossings.
    reversal_threshold = min(0.90, max(0.68, entry_threshold + 0.08))
    opposite_qualified = bool(
        candidate == opposite_candidate
        and fusion.get("paper_entry_allowed")
        and meta_decision == "eligible"
        and confidence >= reversal_threshold
        and data_quality >= 0.75
    )

    if opposite_qualified:
        result.update(
            {
                "action": "exit",
                "reason": "qualified_opposite_thesis",
                "exit_requested": True,
                "exit_reason": "thesis_reversal",
                "opposite_candidate": candidate,
                "confidence": round(confidence, 4),
                "reversal_threshold": round(reversal_threshold, 4),
                "data_quality": round(data_quality, 4),
            }
        )
        if protective_stop != current_stop:
            result["new_stop_loss"] = round(protective_stop, 4)
        return result

    if protective_stop != current_stop:
        result.update(
            {
                "action": "tighten_stop",
                "reason": stop_reason,
                "new_stop_loss": round(protective_stop, 4),
            }
        )
    return result


def _confirm_reversal_exit(
    position_key: str,
    management: dict | None,
    streaks: dict[str, dict[str, object]],
    *,
    observation_id: str | None = None,
    required: int = 2,
) -> dict | None:
    """Require consecutive qualified opposite-thesis observations before exit.

    A single noisy reversal signal is never sufficient. Any interruption resets
    the streak. This state is deliberately in-memory: after a process restart,
    the engine must reconfirm rather than trust stale reversal evidence.
    """
    key = str(position_key or "").strip()
    required = max(2, int(required))

    if management is None:
        if key:
            streaks.pop(key, None)
        return None

    result = dict(management)
    thesis_exit = bool(
        result.get("exit_requested")
        and str(result.get("exit_reason") or "") == "thesis_reversal"
    )

    if not thesis_exit:
        if key:
            streaks.pop(key, None)
        result["confirmation_streak"] = 0
        result["confirmation_required"] = required
        return result

    observation = str(observation_id or "").strip()
    if not observation:
        if key:
            streaks.pop(key, None)
        result["confirmation_streak"] = 0
        result["confirmation_required"] = required
        result["confirmation_observation_missing"] = True
        result["proposed_exit_reason"] = "thesis_reversal"
        result["exit_requested"] = False
        result["exit_reason"] = None
        result["action"] = "hold"
        result["reason"] = "opposite_thesis_missing_observation"
        return result

    prior = streaks.get(key, {}) if key else {}
    prior_count = int(prior.get("count", 0) or 0) if isinstance(prior, dict) else 0
    prior_observation = (
        str(prior.get("observation_id") or "")
        if isinstance(prior, dict)
        else ""
    )

    if observation and observation == prior_observation:
        streak = prior_count
        result["confirmation_streak"] = streak
        result["confirmation_required"] = required
        result["confirmation_observation_reused"] = True
        result["proposed_exit_reason"] = "thesis_reversal"
        result["exit_requested"] = False
        result["exit_reason"] = None
        result["action"] = "hold"
        result["reason"] = "opposite_thesis_waiting_new_observation"
        return result

    streak = prior_count + 1
    if key:
        streaks[key] = {
            "count": streak,
            "observation_id": observation,
        }
    result["confirmation_streak"] = streak
    result["confirmation_required"] = required
    result["confirmation_observation_reused"] = False

    if streak < required:
        result["proposed_exit_reason"] = "thesis_reversal"
        result["exit_requested"] = False
        result["exit_reason"] = None
        result["action"] = "hold"
        result["reason"] = "opposite_thesis_confirmation_pending"
        return result

    if key:
        streaks.pop(key, None)
    result["reason"] = "confirmed_opposite_thesis"
    return result


def _paper_exit_fill_price(
    side: str,
    mark: float,
    stop_loss: float,
    target_price: float,
    exit_reason: str,
) -> float:
    """Conservative paper fill once a stop/target condition is observed.

    Targets fill at the target level instead of crediting favorable overshoot.
    Stops preserve adverse gap/slippage by taking the worse observed mark.
    """
    if exit_reason == "target_price":
        return float(target_price)
    if exit_reason == "stop_loss":
        if side == "long":
            return min(float(mark), float(stop_loss))
        return max(float(mark), float(stop_loss))
    return float(mark)


def _serialize_account(account: XAUPaperAccount | None) -> dict | None:
    if not account:
        return None
    return {
        "id": account.id,
        "week_key": account.week_key,
        "initial_capital": account.initial_capital,
        "realized_pnl": account.realized_pnl,
        "current_equity": account.current_equity,
        "peak_equity": account.peak_equity,
        "max_drawdown_pct": account.max_drawdown_pct,
        "total_trades": account.total_trades,
        "winning_trades": account.winning_trades,
        "losing_trades": account.losing_trades,
        "status": account.status,
        "started_at": account.started_at.isoformat() if account.started_at else None,
        "ended_at": account.ended_at.isoformat() if account.ended_at else None,
    }


def _serialize_position(position: XAUPaperPosition | None) -> dict | None:
    if not position:
        return None
    return {
        "id": position.id,
        "account_id": position.account_id,
        "setup_key": position.setup_key,
        "side": position.side,
        "quantity_oz": position.quantity_oz,
        "entry_price": position.entry_price,
        "stop_loss": position.stop_loss,
        "target_price": position.target_price,
        "current_price": position.current_price,
        "unrealized_pnl": position.unrealized_pnl,
        "mfe_usd": position.mfe_usd,
        "mae_usd": position.mae_usd,
        "risk_usd": position.risk_usd,
        "setup_state": position.setup_state,
        "macro_relation": position.macro_relation,
        "price_source": position.price_source,
        "status": position.status,
        "opened_at": position.opened_at.isoformat() if position.opened_at else None,
        "closed_at": position.closed_at.isoformat() if position.closed_at else None,
    }


def _serialize_trade(trade: XAUPaperTrade) -> dict:
    return {
        "id": trade.id,
        "account_id": trade.account_id,
        "setup_key": trade.setup_key,
        "side": trade.side,
        "quantity_oz": trade.quantity_oz,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "stop_loss": trade.stop_loss,
        "target_price": trade.target_price,
        "pnl": trade.pnl,
        "pnl_pct_equity": trade.pnl_pct_equity,
        "r_multiple": trade.r_multiple,
        "mfe_usd": trade.mfe_usd,
        "mae_usd": trade.mae_usd,
        "risk_usd": trade.risk_usd,
        "exit_reason": trade.exit_reason,
        "setup_state": trade.setup_state,
        "macro_relation": trade.macro_relation,
        "price_source": trade.price_source,
        "opened_at": trade.opened_at.isoformat() if trade.opened_at else None,
        "closed_at": trade.closed_at.isoformat() if trade.closed_at else None,
        "meta": trade.meta or {},
    }


def _performance_metrics(trades: list[XAUPaperTrade]) -> dict:
    if not trades:
        return {
            "trade_count": 0,
            "average_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": None,
            "average_mfe_usd": 0.0,
            "average_mae_usd": 0.0,
            "average_win_r": 0.0,
            "average_loss_r": 0.0,
            "win_rate": 0.0,
        }

    r_values = [float(item.r_multiple or 0.0) for item in trades]
    wins = [value for value in r_values if value > 0]
    losses = [value for value in r_values if value < 0]
    gross_profit = sum(max(float(item.pnl or 0.0), 0.0) for item in trades)
    gross_loss = abs(sum(min(float(item.pnl or 0.0), 0.0) for item in trades))

    return {
        "trade_count": len(trades),
        "average_r": round(sum(r_values) / len(r_values), 4),
        "expectancy_r": round(sum(r_values) / len(r_values), 4),
        "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else None,
        "average_mfe_usd": round(
            sum(float(item.mfe_usd or 0.0) for item in trades) / len(trades),
            4,
        ),
        "average_mae_usd": round(
            sum(float(item.mae_usd or 0.0) for item in trades) / len(trades),
            4,
        ),
        "average_win_r": round(sum(wins) / len(wins), 4) if wins else 0.0,
        "average_loss_r": round(sum(losses) / len(losses), 4) if losses else 0.0,
        "win_rate": round(len(wins) / len(trades), 4),
    }



def _calibration_metrics(predictions: list[tuple[float, int]]) -> dict:
    """Brier score + 5-bin ECE for historical confidence/outcome pairs."""
    if not predictions:
        return {
            "calibration_sample_count": 0,
            "brier_score": None,
            "expected_calibration_error": None,
        }

    clean = [
        (max(0.0, min(1.0, float(probability))), 1 if int(outcome) > 0 else 0)
        for probability, outcome in predictions
    ]
    brier = sum((probability - outcome) ** 2 for probability, outcome in clean) / len(clean)

    bins: list[list[tuple[float, int]]] = [[] for _ in range(5)]
    for probability, outcome in clean:
        index = min(4, int(probability * 5))
        bins[index].append((probability, outcome))

    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_probability = sum(item[0] for item in bucket) / len(bucket)
        observed_rate = sum(item[1] for item in bucket) / len(bucket)
        ece += (len(bucket) / len(clean)) * abs(avg_probability - observed_rate)

    return {
        "calibration_sample_count": len(clean),
        "brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
    }


def _trade_autopsy(
    position,
    signal_meta: dict,
    *,
    exit_reason: str,
    pnl: float,
    r_multiple: float,
) -> dict:
    """Produce diagnostic attribution. Labels are hypotheses, not causal proof."""
    risk = max(0.0, float(getattr(position, "risk_usd", 0.0) or 0.0))
    mfe = float(getattr(position, "mfe_usd", 0.0) or 0.0)
    mae = float(getattr(position, "mae_usd", 0.0) or 0.0)
    mfe_r = mfe / risk if risk > 0 else 0.0
    mae_r = mae / risk if risk > 0 else 0.0

    cognition = signal_meta.get("cognition") or {}
    confidence = cognition.get("confidence") or {}
    regime = cognition.get("regime") or {}
    hypotheses = cognition.get("hypotheses") or []
    adversarial = cognition.get("adversarial") or {}
    predicted_confidence = _number(
        signal_meta.get("cognitive_confidence"),
        _number(confidence.get("calibrated_confidence"), 0.5),
    )
    primary_hypothesis = (
        str((hypotheses[0] or {}).get("name") or "unknown")
        if hypotheses
        else "unknown"
    )

    outcome = "win" if r_multiple > 0.05 else "loss" if r_multiple < -0.05 else "flat"
    attributions: list[str] = []

    if outcome == "win":
        attributions.append("thesis_confirmed")
        if mfe_r > 0 and r_multiple / mfe_r < 0.45:
            attributions.append("low_profit_capture")
    else:
        if exit_reason == "time_stop":
            attributions.append("no_follow_through")
        elif exit_reason == "stop_loss" and mfe_r >= 0.50:
            attributions.append("entry_timing_or_stop_too_tight")
        elif exit_reason == "stop_loss":
            attributions.append("directional_thesis_failed")
        elif exit_reason == "thesis_reversal":
            attributions.append("thesis_invalidated_before_hard_stop")
        else:
            attributions.append("setup_failed")

        if regime.get("label") in {"transition", "range_rotation"} and primary_hypothesis == "trend_continuation":
            attributions.append("regime_misclassification_candidate")
        if adversarial.get("counter_evidence"):
            attributions.append("counter_evidence_present_at_entry")
        if predicted_confidence >= 0.72:
            attributions.append("high_confidence_error")

    calibration_outcome = 1 if outcome == "win" else 0
    prediction_error = abs(predicted_confidence - calibration_outcome)
    return {
        "diagnostic_only": True,
        "outcome": outcome,
        "primary_attribution": attributions[0] if attributions else "unclassified",
        "attributions": list(dict.fromkeys(attributions)),
        "predicted_confidence": round(max(0.0, min(1.0, predicted_confidence)), 4),
        "calibration_outcome": calibration_outcome,
        "prediction_error": round(prediction_error, 4),
        "regime": regime.get("label"),
        "primary_hypothesis": primary_hypothesis,
        "mfe_r": round(mfe_r, 4),
        "mae_r": round(mae_r, 4),
        "realized_r": round(r_multiple, 4),
        "pnl": round(pnl, 4),
        "exit_reason": exit_reason,
    }


def _autopsy_counts(trades: list[XAUPaperTrade]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for trade in trades:
        autopsy = (trade.meta or {}).get("autopsy") or {}
        label = str(autopsy.get("primary_attribution") or "").strip()
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def _trade_diagnostics(trades: list[XAUPaperTrade]) -> dict:
    predictions: list[tuple[float, int]] = []
    autopsied = 0
    for trade in trades:
        autopsy = (trade.meta or {}).get("autopsy") or {}
        predicted = autopsy.get("predicted_confidence")
        outcome = autopsy.get("calibration_outcome")
        if predicted is not None and outcome is not None:
            predictions.append((float(predicted), int(outcome)))
            autopsied += 1

    calibration = _calibration_metrics(predictions)
    return {
        **calibration,
        "autopsied_trades": autopsied,
        "autopsy_counts": _autopsy_counts(trades),
    }


def _shadow_horizon_due(elapsed_minutes: float, horizon_minutes: int) -> tuple[bool, float]:
    """Return whether a shadow horizon can be measured without hindsight drift."""
    horizon = max(1.0, float(horizon_minutes))
    grace = max(2.0, min(15.0, horizon * 0.10))
    elapsed = max(0.0, float(elapsed_minutes))
    return horizon <= elapsed <= horizon + grace, grace


def _shadow_metrics(signals: list[XAUPaperSignal]) -> dict:
    """Aggregate counterfactual outcomes for accepted and rejected setups."""
    buckets: dict[str, dict[str, list[float]]] = {}
    for signal in signals:
        candidate = str(signal.candidate or "")
        if candidate not in {"long_setup", "short_setup"}:
            continue
        decision = "accepted" if bool(signal.accepted) else "rejected"
        shadow = (signal.meta or {}).get("shadow") or {}
        for horizon, payload in shadow.items():
            if not isinstance(payload, dict):
                continue
            value = _number(payload.get("directional_return_bps"))
            if value is None:
                continue
            bucket = buckets.setdefault(
                str(horizon),
                {"accepted": [], "rejected": []},
            )
            bucket[decision].append(float(value))

    out: dict[str, dict] = {}
    for horizon, groups in buckets.items():
        out[horizon] = {}
        for decision, values in groups.items():
            if not values:
                out[horizon][decision] = {
                    "count": 0,
                    "positive_rate": None,
                    "average_directional_return_bps": None,
                }
                continue
            positives = sum(1 for value in values if value > 0)
            out[horizon][decision] = {
                "count": len(values),
                "positive_rate": round(positives / len(values), 4),
                "average_directional_return_bps": round(sum(values) / len(values), 4),
            }
    return out


def _shadow_research_memory(
    signals: list[XAUPaperSignal],
    current_vector: dict,
    *,
    similarity_floor: float = 0.68,
    limit: int = 80,
) -> dict:
    """Research-only episodic memory from forward shadow outcomes.

    Shadow outcomes are never treated as executed trades, never feed Brier/PnL
    calibration, and only provide a small contextual prior in cognition.
    """
    eligible_rejections = {
        "cognitive_veto",
        "cognitive_wait",
        "cognitive_observe",
        "setup_macro_conflict",
    }
    episodes: list[tuple[float, float, str]] = []
    for signal in signals:
        # Shadow learning is only for false-negative analysis of epistemic
        # decisions. Executed/accepted setups and operational gates such as
        # spread, market closure, position_already_open, data/event gates must
        # never train confidence.
        if bool(getattr(signal, "accepted", False)):
            continue
        rejection_reason = str(
            getattr(signal, "rejection_reason", "") or ""
        )
        if rejection_reason not in eligible_rejections:
            continue

        meta = signal.meta or {}
        shadow = meta.get("shadow") or {}
        if not shadow:
            continue
        saved_vector = (
            meta.get("state_vector")
            or ((meta.get("cognition") or {}).get("market_state"))
            or {}
        )
        similarity = state_vector_similarity(current_vector, saved_vector)
        if similarity < similarity_floor:
            continue

        payload = None
        horizon = None
        for key in ("60m", "30m", "15m"):
            candidate_payload = shadow.get(key)
            if isinstance(candidate_payload, dict) and _number(
                candidate_payload.get("directional_return_bps")
            ) is not None:
                payload = candidate_payload
                horizon = key
                break
        if payload is None or horizon is None:
            continue

        episodes.append(
            (
                similarity,
                float(payload.get("directional_return_bps")),
                horizon,
            )
        )

    episodes.sort(key=lambda item: item[0], reverse=True)
    episodes = episodes[: max(1, int(limit))]
    if not episodes:
        return {
            "source": "shadow_research_only",
            "sample_count": 0,
            "positive_rate": None,
            "average_directional_return_bps": None,
            "similarity_weighted_return_bps": None,
            "average_similarity": None,
            "nearest_similarity": None,
            "horizon_mix": {},
            "decision_filtered": True,
            "eligible_rejection_reasons": sorted(eligible_rejections),
            "research_only": True,
        }

    total_weight = sum(item[0] for item in episodes) or 1.0
    positive_rate = sum(1 for _, value, _ in episodes if value > 0) / len(episodes)
    average_return = sum(value for _, value, _ in episodes) / len(episodes)
    weighted_return = sum(sim * value for sim, value, _ in episodes) / total_weight
    horizon_mix: dict[str, int] = {}
    for _, _, horizon in episodes:
        horizon_mix[horizon] = horizon_mix.get(horizon, 0) + 1

    return {
        "source": "shadow_research_only",
        "sample_count": len(episodes),
        "positive_rate": round(positive_rate, 4),
        "average_directional_return_bps": round(average_return, 4),
        "similarity_weighted_return_bps": round(weighted_return, 4),
        "average_similarity": round(
            sum(item[0] for item in episodes) / len(episodes),
            4,
        ),
        "nearest_similarity": round(episodes[0][0], 4),
        "horizon_mix": horizon_mix,
        "decision_filtered": True,
        "eligible_rejection_reasons": sorted(eligible_rejections),
        "research_only": True,
    }


def _replay_research_memory(
    episodes: list[object],
    current_vector: dict,
    *,
    similarity_floor: float = 0.70,
    limit: int = 120,
) -> dict:
    """Research-only memory retrieved from no-lookahead replay episodes."""
    scored: list[tuple[float, float]] = []
    for episode in episodes:
        meta = getattr(episode, "meta", {}) or {}
        replay_payload = meta.get("replay_episode") or {}
        saved_vector = (
            getattr(episode, "state_vector", None)
            or meta.get("state_vector")
            or replay_payload.get("state_vector")
            or {}
        )
        directional_return = getattr(episode, "directional_return_bps", None)
        if directional_return is None:
            directional_return = (
                meta.get("directional_return_bps")
                if meta.get("directional_return_bps") is not None
                else replay_payload.get("directional_return_bps")
            )
        if directional_return is None:
            continue
        similarity = state_vector_similarity(current_vector, saved_vector)
        if similarity < similarity_floor:
            continue
        scored.append((similarity, float(directional_return)))

    scored.sort(key=lambda item: item[0], reverse=True)
    scored = scored[: max(1, int(limit))]
    if not scored:
        return {
            "source": "walk_forward_replay",
            "sample_count": 0,
            "positive_rate": None,
            "average_directional_return_bps": None,
            "similarity_weighted_return_bps": None,
            "average_similarity": None,
            "nearest_similarity": None,
            "research_only": True,
            "lookahead_protected": True,
        }

    total_weight = sum(item[0] for item in scored) or 1.0
    return {
        "source": "walk_forward_replay",
        "sample_count": len(scored),
        "positive_rate": round(
            sum(1 for _, value in scored if value > 0) / len(scored),
            4,
        ),
        "average_directional_return_bps": round(
            sum(value for _, value in scored) / len(scored),
            4,
        ),
        "similarity_weighted_return_bps": round(
            sum(similarity * value for similarity, value in scored) / total_weight,
            4,
        ),
        "average_similarity": round(
            sum(item[0] for item in scored) / len(scored),
            4,
        ),
        "nearest_similarity": round(scored[0][0], 4),
        "research_only": True,
        "lookahead_protected": True,
    }


def _serialize_signal(signal: XAUPaperSignal) -> dict:
    return {
        "id": signal.id,
        "account_id": signal.account_id,
        "setup_key": signal.setup_key,
        "candidate": signal.candidate,
        "fusion_state": signal.fusion_state,
        "macro_relation": signal.macro_relation,
        "event_risk": signal.event_risk,
        "price": signal.price,
        "accepted": signal.accepted,
        "rejection_reason": signal.rejection_reason,
        "observed_at": signal.observed_at.isoformat() if signal.observed_at else None,
        "meta": signal.meta or {},
    }


class XAUPaperTradingEngine:
    """Single-position, weekly-reset XAU paper league."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._reversal_streaks: dict[str, dict[str, object]] = {}

    def _active_account(self, db) -> XAUPaperAccount | None:
        return (
            db.query(XAUPaperAccount)
            .filter(XAUPaperAccount.status == "active")
            .order_by(XAUPaperAccount.id.desc())
            .first()
        )

    def _open_position(self, db, account_id: int | None = None) -> XAUPaperPosition | None:
        query = db.query(XAUPaperPosition).filter(XAUPaperPosition.status == "open")
        if account_id is not None:
            query = query.filter(XAUPaperPosition.account_id == account_id)
        return query.order_by(XAUPaperPosition.id.desc()).first()

    def _memory_snapshot(
        self,
        db,
        technical: dict,
        macro: dict,
    ) -> dict:
        """Retrieve state-vector-nearest episodes and calibration history."""
        candidate = str(technical.get("candidate") or "none")
        current_vector = build_market_state_vector(technical, macro)

        signals = (
            db.query(XAUPaperSignal)
            .filter(
                XAUPaperSignal.accepted == True,  # noqa: E712
                XAUPaperSignal.candidate == candidate,
            )
            .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
            .limit(500)
            .all()
        )
        setup_keys = [signal.setup_key for signal in signals]
        trades = []
        if setup_keys:
            trades = (
                db.query(XAUPaperTrade)
                .filter(XAUPaperTrade.setup_key.in_(setup_keys))
                .order_by(XAUPaperTrade.closed_at.desc(), XAUPaperTrade.id.desc())
                .limit(500)
                .all()
            )
        trade_by_key = {trade.setup_key: trade for trade in trades}

        scored: list[tuple[float, XAUPaperSignal, XAUPaperTrade]] = []
        calibration_predictions: list[tuple[float, int]] = []
        for signal in signals:
            trade = trade_by_key.get(signal.setup_key)
            if trade is None:
                continue
            meta = signal.meta or {}
            saved_vector = (
                meta.get("state_vector")
                or ((meta.get("cognition") or {}).get("market_state"))
                or {}
            )
            similarity = state_vector_similarity(current_vector, saved_vector)

            # Backward-compatible fallback for episodes recorded before v2 vectors.
            if similarity <= 0.0:
                similarity = 0.35
                if str(meta.get("alignment") or "") == str(current_vector.get("alignment") or ""):
                    similarity += 0.15
                if str(meta.get("regime") or "") == str(current_vector.get("regime") or ""):
                    similarity += 0.15
                if str((meta.get("micro") or {}).get("direction") or "") in str(current_vector.get("candidate") or ""):
                    similarity += 0.05
                similarity = min(0.70, similarity)

            scored.append((similarity, signal, trade))

            predicted = meta.get("cognitive_confidence")
            if predicted is None:
                predicted = (
                    ((meta.get("cognition") or {}).get("confidence") or {})
                    .get("calibrated_confidence")
                )
            if predicted is not None:
                calibration_predictions.append(
                    (float(predicted), 1 if float(trade.r_multiple or 0.0) > 0.05 else 0)
                )

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [item for item in scored[:60] if item[0] >= 0.68]
        selected_trades = [item[2] for item in selected]

        if selected_trades:
            memory_trades = selected_trades
            source = "state_vector_similar_closed_setups"
            similarities = [item[0] for item in selected]
            similar_samples = len(selected_trades)
        else:
            memory_trades = (
                db.query(XAUPaperTrade)
                .order_by(XAUPaperTrade.closed_at.desc(), XAUPaperTrade.id.desc())
                .limit(120)
                .all()
            )
            source = "global_closed_paper_trades_fallback"
            similarities = []
            similar_samples = 0

        shadow_signals = (
            db.query(XAUPaperSignal)
            .filter(XAUPaperSignal.candidate == candidate)
            .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
            .limit(500)
            .all()
        )
        shadow_memory = _shadow_research_memory(
            shadow_signals,
            current_vector,
        )

        if replay_store_is_external():
            replay_db = open_xau_replay_session()
            try:
                replay_episodes = (
                    replay_db.query(XAUReplayEpisode)
                    .filter(XAUReplayEpisode.candidate == candidate)
                    .order_by(
                        XAUReplayEpisode.observed_at.desc(),
                        XAUReplayEpisode.id.desc(),
                    )
                    .limit(1000)
                    .all()
                )
            except Exception:
                replay_episodes = []
            finally:
                replay_db.close()
        else:
            replay_episodes = (
                db.query(XAUPaperSignal)
                .filter(
                    XAUPaperSignal.candidate == f"replay_{candidate}",
                    XAUPaperSignal.rejection_reason == "historical_replay",
                )
                .order_by(
                    XAUPaperSignal.observed_at.desc(),
                    XAUPaperSignal.id.desc(),
                )
                .limit(1000)
                .all()
            )

        replay_memory = _replay_research_memory(
            replay_episodes,
            current_vector,
        )

        metrics = _performance_metrics(memory_trades)
        wins = sum(1 for trade in memory_trades if float(trade.r_multiple or 0.0) > 0.05)
        sample_count = len(memory_trades)
        empirical = (wins / sample_count) if sample_count else None
        # Beta(2,2) prior prevents small-sample confidence from becoming extreme.
        posterior = ((wins + 2.0) / (sample_count + 4.0)) if sample_count else None

        calibration = _calibration_metrics(calibration_predictions)
        metrics.update(calibration)
        metrics.update(
            {
                "source": source,
                "similar_samples": similar_samples,
                "candidate": candidate,
                "alignment": current_vector.get("alignment"),
                "regime": current_vector.get("regime"),
                "session": current_vector.get("session"),
                "empirical_win_rate": round(empirical, 4) if empirical is not None else None,
                "posterior_win_probability": round(posterior, 4) if posterior is not None else None,
                "average_similarity": (
                    round(sum(similarities) / len(similarities), 4)
                    if similarities
                    else None
                ),
                "nearest_similarity": round(similarities[0], 4) if similarities else None,
                "current_state_vector": current_vector,
                "autopsy_counts": _autopsy_counts(memory_trades),
                "shadow_memory": shadow_memory,
                "replay_memory": replay_memory,
            }
        )

        if selected:
            weighted_denominator = sum(item[0] for item in selected) or 1.0
            metrics["similarity_weighted_expectancy_r"] = round(
                sum(float(item[2].r_multiple or 0.0) * item[0] for item in selected)
                / weighted_denominator,
                4,
            )
        else:
            metrics["similarity_weighted_expectancy_r"] = None
        return metrics

    def _update_shadow_outcomes(
        self,
        db,
        *,
        reference_price: float | None,
        now_utc: datetime,
        reference_source: str = "",
    ) -> int:
        """Journal forward outcomes for historical setup decisions.

        This never opens/closes positions. It only measures what happened after
        both accepted and rejected directional setups so false negatives can be
        evaluated later.
        """
        price = _number(reference_price)
        if price is None:
            return 0

        cutoff = now_utc - timedelta(hours=6)
        signals = (
            db.query(XAUPaperSignal)
            .filter(
                XAUPaperSignal.observed_at >= cutoff,
                XAUPaperSignal.candidate.in_(("long_setup", "short_setup")),
            )
            .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
            .limit(500)
            .all()
        )

        horizons = (15, 30, 60, 240)
        changed = 0
        for signal in signals:
            if not signal.observed_at or not signal.price:
                continue
            elapsed = _position_age_minutes(signal.observed_at, now_utc)
            side = 1.0 if signal.candidate == "long_setup" else -1.0
            meta = dict(signal.meta or {})
            shadow = dict(meta.get("shadow") or {})
            touched = False

            for horizon in horizons:
                key = f"{horizon}m"
                if key in shadow or elapsed < horizon:
                    continue

                # Counterfactuals are only valid near their target horizon.
                # Never backfill a 15m outcome with the price observed hours
                # later after a restart/outage; missing evidence is preferable
                # to hindsight-contaminated learning data.
                due, grace_minutes = _shadow_horizon_due(elapsed, horizon)
                if not due:
                    continue

                directional_return_bps = (
                    ((price - float(signal.price)) / float(signal.price))
                    * 10_000.0
                    * side
                )
                shadow[key] = {
                    "directional_return_bps": round(directional_return_bps, 4),
                    "positive": directional_return_bps > 0,
                    "reference_price": round(price, 6),
                    "reference_source": reference_source,
                    "target_horizon_minutes": horizon,
                    "observed_after_minutes": round(elapsed, 2),
                    "timing_error_minutes": round(elapsed - horizon, 2),
                    "grace_minutes": round(grace_minutes, 2),
                    "measured_at": now_utc.isoformat(),
                }
                touched = True

            if touched:
                meta["shadow"] = shadow
                signal.meta = meta
                changed += 1

        if changed:
            db.flush()
        return changed

    def _update_account_equity(
        self,
        account: XAUPaperAccount,
        unrealized_pnl: float = 0.0,
        *,
        track_extremes: bool = True,
    ) -> None:
        equity = float(account.initial_capital) + float(account.realized_pnl) + float(unrealized_pnl)
        account.current_equity = round(equity, 4)
        if not track_extremes:
            return
        account.peak_equity = max(float(account.peak_equity or account.initial_capital), equity)
        if account.peak_equity > 0:
            dd = max(0.0, (account.peak_equity - equity) / account.peak_equity * 100.0)
            account.max_drawdown_pct = max(float(account.max_drawdown_pct or 0.0), dd)

    def _close_position(
        self,
        db,
        account: XAUPaperAccount,
        position: XAUPaperPosition,
        exit_price: float,
        exit_reason: str,
        now_utc: datetime,
        meta: dict | None = None,
    ) -> XAUPaperTrade:
        pnl = _pnl(position.side, position.entry_price, exit_price, position.quantity_oz)
        pnl = round(pnl, 4)
        initial = float(account.initial_capital or 0.0)
        pnl_pct = (pnl / initial * 100.0) if initial > 0 else 0.0
        risk = float(position.risk_usd or 0.0)
        r_multiple = (pnl / risk) if risk > 0 else 0.0

        trade_meta = dict(meta or {})
        trade_meta.setdefault("engine_version", PAPER_ENGINE_VERSION)

        signal = (
            db.query(XAUPaperSignal)
            .filter(XAUPaperSignal.setup_key == position.setup_key)
            .first()
        )
        signal_meta = dict(signal.meta or {}) if signal else {}
        trade_meta["entry_state_vector"] = signal_meta.get("state_vector")
        trade_meta["entry_cognition"] = signal_meta.get("cognition") or {}
        trade_meta["autopsy"] = _trade_autopsy(
            position,
            signal_meta,
            exit_reason=exit_reason,
            pnl=pnl,
            r_multiple=r_multiple,
        )
        if position.opened_at:
            trade_meta["holding_minutes"] = round(
                _position_age_minutes(position.opened_at, now_utc),
                2,
            )

        trade = XAUPaperTrade(
            account_id=account.id,
            setup_key=position.setup_key,
            side=position.side,
            quantity_oz=position.quantity_oz,
            entry_price=position.entry_price,
            exit_price=exit_price,
            stop_loss=position.stop_loss,
            target_price=position.target_price,
            pnl=pnl,
            pnl_pct_equity=round(pnl_pct, 4),
            r_multiple=round(r_multiple, 4),
            mfe_usd=position.mfe_usd,
            mae_usd=position.mae_usd,
            risk_usd=position.risk_usd,
            exit_reason=exit_reason,
            setup_state=position.setup_state,
            macro_relation=position.macro_relation,
            price_source=position.price_source,
            opened_at=position.opened_at,
            closed_at=now_utc,
            meta=trade_meta,
        )
        db.add(trade)

        position.status = "closed"
        position.closed_at = now_utc
        position.current_price = exit_price
        position.unrealized_pnl = pnl

        account.realized_pnl = round(float(account.realized_pnl or 0.0) + pnl, 4)
        account.total_trades = int(account.total_trades or 0) + 1
        if pnl > 0:
            account.winning_trades = int(account.winning_trades or 0) + 1
        elif pnl < 0:
            account.losing_trades = int(account.losing_trades or 0) + 1
        self._update_account_equity(account, 0.0)

        logger.info(
            "[XAU paper] close side=%s qty=%.4f entry=%.2f exit=%.2f pnl=%.2f R=%.2f reason=%s",
            position.side,
            position.quantity_oz,
            position.entry_price,
            exit_price,
            pnl,
            r_multiple,
            exit_reason,
        )
        return trade

    def _ensure_week(
        self,
        db,
        spot: dict | None,
        now: datetime | None = None,
    ) -> XAUPaperAccount:
        key = _week_key(self.settings, now)
        active = self._active_account(db)
        now_utc = _utc_naive(now)

        if active and active.week_key == key:
            return active

        if active:
            old_position = self._open_position(db, active.id)
            if old_position:
                spot = spot or {}
                # Never manufacture a weekly-reset fill from a stale/mid-only
                # reference. Keep the old weekly account active until a fresh
                # executable-side paper quote becomes available.
                exit_quote = _weekly_reset_fill_price(old_position.side, spot)
                if exit_quote is None:
                    logger.info(
                        "[XAU paper] weekly rollover deferred: no fresh bid/ask fill old_week=%s new_week=%s",
                        active.week_key,
                        key,
                    )
                    return active

                source = str(spot.get("source") or old_position.price_source or "fill_quote")
                self._close_position(
                    db,
                    active,
                    old_position,
                    float(exit_quote),
                    "weekly_reset",
                    now_utc,
                    {
                        "reset_week": key,
                        "mark_source": source,
                        "fresh_spot": True,
                    },
                )
            active.status = "closed"
            active.ended_at = now_utc
            self._update_account_equity(active, 0.0)

        existing = (
            db.query(XAUPaperAccount)
            .filter(XAUPaperAccount.week_key == key)
            .first()
        )
        if existing:
            existing.status = "active"
            existing.ended_at = None
            db.flush()
            return existing

        capital = float(self.settings.xau_paper_initial_capital)
        account = XAUPaperAccount(
            week_key=key,
            initial_capital=capital,
            realized_pnl=0.0,
            current_equity=capital,
            peak_equity=capital,
            max_drawdown_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            status="active",
            started_at=_week_start_utc_naive(self.settings, now),
        )
        db.add(account)
        db.flush()
        logger.info("[XAU paper] new weekly league %s capital=%.2f", key, capital)
        return account

    def _mark_position(
        self,
        account: XAUPaperAccount,
        position: XAUPaperPosition,
        spot: dict,
        analysis_reference: dict | None = None,
    ) -> float | None:
        mark, mark_kind = _paper_context_mark_price(
            position.side,
            spot,
            analysis_reference,
        )
        if mark is None:
            return None
        pnl = _pnl(position.side, position.entry_price, mark, position.quantity_oz)
        position.current_price = mark
        position.unrealized_pnl = round(pnl, 4)

        if mark_kind == "fill_quote":
            position.mfe_usd = max(float(position.mfe_usd or 0.0), pnl)
            position.mae_usd = min(float(position.mae_usd or 0.0), pnl)
            self._update_account_equity(account, pnl, track_extremes=True)
        else:
            # Contextual MTM is useful for display, but must not contaminate
            # execution-observed MFE/MAE or max-drawdown statistics.
            self._update_account_equity(account, pnl, track_extremes=False)
            logger.debug(
                "[XAU paper] contextual mark updates display equity only; execution metrics remain frozen"
            )
        return mark

    def _levels_and_size(
        self,
        account: XAUPaperAccount,
        side: str,
        entry: float,
        technical: dict,
    ) -> tuple[float, float, float, float] | None:
        atr = _number(technical.get("atr_reference"))
        swing_low = _number(technical.get("swing_low_reference"))
        swing_high = _number(technical.get("swing_high_reference"))

        if side == "long":
            stop = swing_low if swing_low and swing_low < entry else None
            if stop is None and atr and atr > 0:
                stop = entry - atr
            if stop is None or stop <= 0 or stop >= entry:
                return None
            distance = entry - stop
            target = entry + float(self.settings.xau_paper_reward_risk) * distance
        else:
            stop = swing_high if swing_high and swing_high > entry else None
            if stop is None and atr and atr > 0:
                stop = entry + atr
            if stop is None or stop <= entry:
                return None
            distance = stop - entry
            target = entry - float(self.settings.xau_paper_reward_risk) * distance
            if target <= 0:
                return None

        equity = max(0.0, float(account.current_equity or account.initial_capital))
        risk_budget = equity * float(self.settings.xau_paper_risk_pct)
        qty_by_risk = risk_budget / distance if distance > 0 else 0.0
        max_notional = equity * float(self.settings.xau_paper_max_leverage)
        qty_by_notional = max_notional / entry if entry > 0 else 0.0
        quantity = min(qty_by_risk, qty_by_notional)
        quantity = round(max(0.0, quantity), 4)
        if quantity < 0.0001:
            return None

        risk_usd = quantity * distance
        return round(stop, 4), round(target, 4), quantity, round(risk_usd, 4)

    def _setup_key(
        self,
        account: XAUPaperAccount,
        technical: dict,
        fusion: dict,
    ) -> str:
        frames = technical.get("frames") or {}
        anchor = (
            (frames.get("15m") or {}).get("observed_at")
            or (technical.get("micro") or {}).get("last_point_at")
            or technical.get("observed_at")
            or _utc_naive().isoformat()
        )
        return (
            f"{account.week_key}:"
            f"{fusion.get('technical_candidate')}:"
            f"{anchor}"
        )

    def _record_signal(
        self,
        db,
        account: XAUPaperAccount,
        technical: dict,
        macro: dict,
        fusion: dict,
        spot: dict,
        accepted: bool,
        rejection_reason: str,
        now_utc: datetime,
    ) -> XAUPaperSignal | None:
        setup_key = self._setup_key(account, technical, fusion)
        existing = (
            db.query(XAUPaperSignal)
            .filter(XAUPaperSignal.setup_key == setup_key)
            .first()
        )
        if existing:
            if _can_revalidate_signal(
                bool(existing.accepted),
                str(existing.rejection_reason or ""),
                accepted,
            ):
                previous_reason = existing.rejection_reason
                previous_observed_at = existing.observed_at
                meta = dict(existing.meta or {})
                meta["revalidated"] = True
                meta["initial_rejection_reason"] = previous_reason
                meta["initial_observed_at"] = (
                    previous_observed_at.isoformat()
                    if previous_observed_at
                    else None
                )
                meta["revalidated_at"] = now_utc.isoformat()
                meta["state_vector"] = (
                    ((fusion.get("cognition") or {}).get("market_state"))
                    or build_market_state_vector(technical, macro)
                )
                meta["cognition"] = fusion.get("cognition") or {}
                meta["regime"] = fusion.get("regime")
                meta["cognitive_confidence"] = fusion.get("cognitive_confidence")
                meta["meta_decision"] = fusion.get("meta_decision")
                meta["spot"] = {
                    "price": spot.get("price"),
                    "bid": spot.get("bid"),
                    "ask": spot.get("ask"),
                    "spread_bps": _spot_spread_bps(spot),
                    "source": spot.get("source"),
                    "age_seconds": spot.get("age_seconds"),
                    "is_stale": spot.get("is_stale"),
                }
                existing.accepted = True
                existing.rejection_reason = ""
                existing.fusion_state = str(fusion.get("state") or "")
                existing.macro_relation = str(fusion.get("macro_relation") or "")
                existing.event_risk = bool(fusion.get("event_risk"))
                existing.price = _number(spot.get("price"))
                existing.observed_at = now_utc
                existing.meta = meta
                db.flush()
                logger.info(
                    "[XAU paper] setup revalidated candidate=%s prior_reason=%s price=%s",
                    existing.candidate,
                    previous_reason,
                    existing.price,
                )
            return existing

        signal = XAUPaperSignal(
            account_id=account.id,
            setup_key=setup_key,
            candidate=str(fusion.get("technical_candidate") or "none"),
            fusion_state=str(fusion.get("state") or ""),
            macro_relation=str(fusion.get("macro_relation") or ""),
            event_risk=bool(fusion.get("event_risk")),
            price=_number(spot.get("price")),
            accepted=accepted,
            rejection_reason=rejection_reason,
            observed_at=now_utc,
            meta={
                "engine_version": PAPER_ENGINE_VERSION,
                "state_vector": (
                    ((fusion.get("cognition") or {}).get("market_state"))
                    or build_market_state_vector(technical, macro)
                ),
                "technical_mode": technical.get("technical_mode"),
                "alignment": technical.get("alignment"),
                "atr_reference": technical.get("atr_reference"),
                "swing_high_reference": technical.get("swing_high_reference"),
                "swing_low_reference": technical.get("swing_low_reference"),
                "frames": {
                    name: {
                        key: frame.get(key)
                        for key in (
                            "direction",
                            "close",
                            "ema_fast",
                            "ema_slow",
                            "rsi14",
                            "atr14",
                            "breakout",
                            "observed_at",
                            "source",
                        )
                    }
                    for name, frame in (technical.get("frames") or {}).items()
                    if isinstance(frame, dict)
                },
                "micro": {
                    key: (technical.get("micro") or {}).get(key)
                    for key in (
                        "direction",
                        "price",
                        "return_10m_pct",
                        "return_30m_pct",
                        "age_seconds",
                        "source",
                    )
                },
                "macro_bias": macro.get("bias"),
                "macro_confidence": macro.get("confidence"),
                "macro_relation": fusion.get("macro_relation"),
                "event_risk": fusion.get("event_risk"),
                "event_kind": fusion.get("event_kind"),
                "event_name": fusion.get("event_name"),
                "event_time_utc": fusion.get("event_time_utc"),
                "event_age_minutes": fusion.get("event_age_minutes"),
                "event_confidence": fusion.get("event_confidence"),
                "event_validation": fusion.get("event_validation"),
                "event_policy_version": fusion.get("event_policy_version"),
                "event_source_url": fusion.get("event_source_url"),
                "fusion_reasons": fusion.get("reasons") or [],
                "cognition": fusion.get("cognition") or {},
                "regime": fusion.get("regime"),
                "cognitive_confidence": fusion.get("cognitive_confidence"),
                "meta_decision": fusion.get("meta_decision"),
                "spot": {
                    "price": spot.get("price"),
                    "bid": spot.get("bid"),
                    "ask": spot.get("ask"),
                    "spread_bps": _spot_spread_bps(spot),
                    "source": spot.get("source"),
                    "age_seconds": spot.get("age_seconds"),
                    "is_stale": spot.get("is_stale"),
                },
                "execution_allowed": False,
            },
        )
        try:
            with db.begin_nested():
                db.add(signal)
                db.flush()
            logger.info(
                "[XAU paper] setup candidate=%s state=%s accepted=%s reason=%s price=%s",
                signal.candidate,
                signal.fusion_state,
                signal.accepted,
                signal.rejection_reason or "accepted",
                signal.price,
            )
            return signal
        except IntegrityError:
            return (
                db.query(XAUPaperSignal)
                .filter(XAUPaperSignal.setup_key == setup_key)
                .first()
            )

    def _try_open(
        self,
        db,
        account: XAUPaperAccount,
        technical: dict,
        macro: dict,
        fusion: dict,
        spot: dict,
        now_utc: datetime,
    ) -> XAUPaperPosition | None:
        candidate = str(fusion.get("technical_candidate") or "none")
        state = str(fusion.get("state") or "")
        existing_position = self._open_position(db, account.id)
        accepted_state, rejection_reason = _entry_gate_reason(
            candidate=candidate,
            fusion_state=state,
            spot=spot,
            has_open_position=existing_position is not None,
            max_spread_bps=float(self.settings.xau_paper_max_spread_bps),
        )
        if candidate not in {"long_setup", "short_setup"}:
            return None

        signal = self._record_signal(
            db,
            account,
            technical,
            macro,
            fusion,
            spot,
            accepted_state,
            rejection_reason,
            now_utc,
        )
        if not signal or not signal.accepted:
            return None

        setup_key = signal.setup_key
        already_traded = (
            db.query(XAUPaperPosition)
            .filter(XAUPaperPosition.setup_key == setup_key)
            .first()
        )
        if already_traded:
            return None

        side = "long" if candidate == "long_setup" else "short"
        entry = _paper_entry_price(side, spot)
        if entry is None:
            signal.accepted = False
            signal.rejection_reason = "bid_ask_unavailable"
            return None

        levels = self._levels_and_size(account, side, entry, technical)
        if levels is None:
            signal.accepted = False
            signal.rejection_reason = "risk_levels_unavailable"
            return None
        stop, target, quantity, risk_usd = levels

        position = XAUPaperPosition(
            account_id=account.id,
            setup_key=setup_key,
            side=side,
            quantity_oz=quantity,
            entry_price=entry,
            stop_loss=stop,
            target_price=target,
            current_price=entry,
            unrealized_pnl=0.0,
            mfe_usd=0.0,
            mae_usd=0.0,
            risk_usd=risk_usd,
            setup_state=state,
            macro_relation=str(fusion.get("macro_relation") or ""),
            price_source=str(spot.get("source") or "indicative_spot"),
            status="open",
            opened_at=now_utc,
        )
        try:
            with db.begin_nested():
                db.add(position)
                db.flush()
        except IntegrityError:
            logger.info("[XAU paper] duplicate setup suppressed: %s", setup_key)
            return None

        mark = _paper_mark_price(side, spot)
        if mark is not None:
            self._mark_position(account, position, spot)

        logger.info(
            "[XAU paper] open side=%s qty=%.4f entry=%.2f stop=%.2f target=%.2f risk=%.2f state=%s",
            side,
            quantity,
            entry,
            stop,
            target,
            risk_usd,
            state,
        )
        return position

    async def eligibility(self) -> dict:
        technical = await get_xau_snapshot(force=False)
        macro = await get_macro_context(force=False)
        spot = technical.get("indicative_spot") or {}

        db = open_xau_paper_session()
        try:
            account = self._active_account(db)
            memory = self._memory_snapshot(db, technical, macro) if account else {}
            fusion = build_decision_fusion(
                technical,
                macro,
                memory=memory,
                min_confidence=float(self.settings.xau_cognition_min_confidence),
            )
            candidate = str(fusion.get("technical_candidate") or "none")
            state = str(fusion.get("state") or "")
            position = self._open_position(db, account.id) if account else None
            eligible, gate_reason = _entry_gate_reason(
                candidate=candidate,
                fusion_state=state,
                spot=spot,
                has_open_position=position is not None,
                max_spread_bps=float(self.settings.xau_paper_max_spread_bps),
            )

            if not bool(fusion.get("paper_entry_allowed")) and candidate in {
                "long_setup",
                "short_setup",
            }:
                eligible = False
                gate_reason = state or "cognition_not_eligible"

            side = (
                "long"
                if candidate == "long_setup"
                else "short"
                if candidate == "short_setup"
                else None
            )
            projected = None
            if eligible and side and account:
                entry = _paper_entry_price(side, spot)
                if entry is None:
                    eligible = False
                    gate_reason = "bid_ask_unavailable"
                else:
                    levels = self._levels_and_size(account, side, entry, technical)
                    if levels is None:
                        eligible = False
                        gate_reason = "risk_levels_unavailable"
                    else:
                        stop, target, quantity, risk_usd = levels
                        projected = {
                            "side": side,
                            "entry_price": entry,
                            "stop_loss": stop,
                            "target_price": target,
                            "quantity_oz": quantity,
                            "risk_usd": risk_usd,
                        }

            spread_bps = _spot_spread_bps(spot)
            return {
                "eligible": eligible,
                "gate_reason": gate_reason or "eligible",
                "candidate": candidate,
                "fusion_state": state,
                "base_state": fusion.get("base_state"),
                "macro_relation": fusion.get("macro_relation"),
                "macro_bias": fusion.get("macro_bias"),
                "macro_confidence": fusion.get("macro_confidence"),
                "regime": fusion.get("regime"),
                "cognitive_confidence": fusion.get("cognitive_confidence"),
                "meta_decision": fusion.get("meta_decision"),
                "cognition": fusion.get("cognition"),
                "memory": memory,
                "event_risk": fusion.get("event_risk"),
                "event_kind": fusion.get("event_kind"),
                "event_name": fusion.get("event_name"),
                "event_time_utc": fusion.get("event_time_utc"),
                "event_age_minutes": fusion.get("event_age_minutes"),
                "event_confidence": fusion.get("event_confidence"),
                "event_validation": fusion.get("event_validation"),
                "event_policy_version": fusion.get("event_policy_version"),
                "event_source_url": fusion.get("event_source_url"),
                "alignment": technical.get("alignment"),
                "micro_direction": (technical.get("micro") or {}).get("direction"),
                "spot": {
                    "price": spot.get("price"),
                    "bid": spot.get("bid"),
                    "ask": spot.get("ask"),
                    "spread_bps": spread_bps,
                    "source": spot.get("source"),
                    "age_seconds": spot.get("age_seconds"),
                    "is_stale": spot.get("is_stale"),
                },
                "projected": projected,
                "position": _serialize_position(position),
                "execution_allowed": False,
            }
        finally:
            db.close()

    async def scan(self, now: datetime | None = None) -> dict:
        if not self.settings.xau_paper_enabled:
            return {"status": "disabled", "execution_allowed": False}

        technical = await get_xau_snapshot(force=False)
        macro = await get_macro_context(force=False)
        spot = technical.get("indicative_spot") or {}
        analysis_reference = technical.get("analysis_reference") or {}
        now_utc = _utc_naive(now)

        db = open_xau_paper_session()
        try:
            account = self._ensure_week(db, spot, now=now)
            analysis_reference = technical.get("analysis_reference") or {}
            shadow_updates = self._update_shadow_outcomes(
                db,
                reference_price=_number(analysis_reference.get("price")),
                now_utc=now_utc,
                reference_source=str(analysis_reference.get("source") or ""),
            )
            memory = self._memory_snapshot(db, technical, macro)
            fusion = build_decision_fusion(
                technical,
                macro,
                memory=memory,
                min_confidence=float(self.settings.xau_cognition_min_confidence),
            )
            position = self._open_position(db, account.id)
            closed_trade = None

            if position:
                self._mark_position(
                    account,
                    position,
                    spot,
                    analysis_reference,
                )

            position_management = None
            if position:
                reversal_key = str(position.setup_key or position.id)
                exit_quote = _paper_management_quote(position.side, spot)
                if exit_quote is None:
                    # Missing/closed fill market interrupts reversal confirmation.
                    self._reversal_streaks.pop(reversal_key, None)
                else:
                    exit_reason = None
                    if position.side == "long":
                        if exit_quote <= position.stop_loss:
                            exit_reason = "stop_loss"
                        elif exit_quote >= position.target_price:
                            exit_reason = "target_price"
                    else:
                        if exit_quote >= position.stop_loss:
                            exit_reason = "stop_loss"
                        elif exit_quote <= position.target_price:
                            exit_reason = "target_price"

                    if exit_reason is None:
                        position_management = _position_guardian(
                            position,
                            fusion,
                            exit_quote,
                        )
                        observation_id = str(
                            (technical.get("micro") or {}).get("last_point_at")
                            or analysis_reference.get("observed_at")
                            or technical.get("observed_at")
                            or ""
                        )
                        position_management = _confirm_reversal_exit(
                            reversal_key,
                            position_management,
                            self._reversal_streaks,
                            observation_id=observation_id,
                            required=2,
                        )
                        new_stop = _number(
                            (position_management or {}).get("new_stop_loss")
                        )
                        if new_stop is not None:
                            old_stop = float(position.stop_loss)
                            position.stop_loss = new_stop
                            logger.info(
                                "[XAU paper] guardian stop side=%s old=%.4f new=%.4f reason=%s current_r=%s",
                                position.side,
                                old_stop,
                                new_stop,
                                (position_management or {}).get("reason"),
                                (position_management or {}).get("current_r"),
                            )
                        if (position_management or {}).get("exit_requested"):
                            exit_reason = str(
                                (position_management or {}).get("exit_reason")
                                or "thesis_reversal"
                            )
                    else:
                        self._reversal_streaks.pop(reversal_key, None)

                    if exit_reason is None and position.opened_at:
                        held_minutes = _position_age_minutes(position.opened_at, now_utc)
                        if held_minutes >= float(self.settings.xau_paper_max_hold_minutes):
                            exit_reason = "time_stop"
                            self._reversal_streaks.pop(reversal_key, None)

                    if exit_reason:
                        self._reversal_streaks.pop(reversal_key, None)
                        exit_fill = _paper_exit_fill_price(
                            position.side,
                            exit_quote,
                            position.stop_loss,
                            position.target_price,
                            exit_reason,
                        )
                        closed_trade = self._close_position(
                            db,
                            account,
                            position,
                            exit_fill,
                            exit_reason,
                            now_utc,
                            {
                                "spot_source": spot.get("source"),
                                "fusion_state": fusion.get("state"),
                                "position_management": position_management,
                            },
                        )
                        position = None

            opened_position = None
            if position is None:
                opened_position = self._try_open(
                    db,
                    account,
                    technical,
                    macro,
                    fusion,
                    spot,
                    now_utc,
                )
                position = opened_position or self._open_position(db, account.id)

            if position:
                mark = self._mark_position(
                    account,
                    position,
                    spot,
                    analysis_reference,
                )
                # If no current mark exists, preserve the last known unrealized
                # equity rather than falsely resetting an open position to flat PnL.
                if mark is None:
                    self._update_account_equity(
                        account,
                        float(position.unrealized_pnl or 0.0),
                    )
            else:
                self._update_account_equity(account, 0.0)

            db.commit()
            db.refresh(account)
            if position:
                db.refresh(position)

            return {
                "status": "ok",
                "week_key": account.week_key,
                "account": _serialize_account(account),
                "position": _serialize_position(position),
                "opened": bool(opened_position),
                "closed_trade": _serialize_trade(closed_trade) if closed_trade else None,
                "fusion": fusion,
                "memory": memory,
                "shadow_updates": shadow_updates,
                "position_management": position_management,
                "execution_allowed": False,
            }
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def summary(self, trade_limit: int = 20, signal_limit: int = 20) -> dict:
        db = open_xau_paper_session()
        try:
            account = self._active_account(db)
            if not account:
                return {
                    "account": None,
                    "position": None,
                    "trades": [],
                    "signals": [],
                    "settings": self.public_settings(),
                    "performance": _performance_metrics([]),
                    "diagnostics": _trade_diagnostics([]),
                    "execution_allowed": False,
                }

            position = self._open_position(db, account.id)
            trades = (
                db.query(XAUPaperTrade)
                .filter(XAUPaperTrade.account_id == account.id)
                .order_by(XAUPaperTrade.closed_at.desc(), XAUPaperTrade.id.desc())
                .limit(max(1, min(int(trade_limit), 200)))
                .all()
            )
            signals = (
                db.query(XAUPaperSignal)
                .filter(
                    XAUPaperSignal.account_id == account.id,
                    XAUPaperSignal.rejection_reason != "historical_replay",
                )
                .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
                .limit(max(1, min(int(signal_limit), 200)))
                .all()
            )
            shadow_signals = (
                db.query(XAUPaperSignal)
                .filter(
                    XAUPaperSignal.account_id == account.id,
                    XAUPaperSignal.rejection_reason != "historical_replay",
                )
                .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
                .limit(500)
                .all()
            )
            return {
                "account": _serialize_account(account),
                "position": _serialize_position(position),
                "trades": [_serialize_trade(item) for item in trades],
                "signals": [_serialize_signal(item) for item in signals],
                "settings": self.public_settings(),
                "performance": _performance_metrics(trades),
                "diagnostics": _trade_diagnostics(trades),
                "shadow_diagnostics": _shadow_metrics(shadow_signals),
                "execution_allowed": False,
            }
        finally:
            db.close()

    def history(self, limit: int = 12) -> list[dict]:
        db = open_xau_paper_session()
        try:
            accounts = (
                db.query(XAUPaperAccount)
                .order_by(XAUPaperAccount.started_at.desc(), XAUPaperAccount.id.desc())
                .limit(max(1, min(int(limit), 52)))
                .all()
            )
            out: list[dict] = []
            for account in accounts:
                trades = (
                    db.query(XAUPaperTrade)
                    .filter(XAUPaperTrade.account_id == account.id)
                    .order_by(XAUPaperTrade.closed_at.asc(), XAUPaperTrade.id.asc())
                    .all()
                )
                item = _serialize_account(account) or {}
                initial = float(account.initial_capital or 0.0)
                equity = float(account.current_equity or initial)
                item["return_pct"] = (
                    round((equity - initial) / initial * 100.0, 4)
                    if initial > 0
                    else 0.0
                )
                item["win_rate"] = (
                    round(float(account.winning_trades or 0) / float(account.total_trades) * 100.0, 2)
                    if account.total_trades
                    else 0.0
                )
                item["performance"] = _performance_metrics(trades)
                item["diagnostics"] = _trade_diagnostics(trades)
                out.append(item)
            return out
        finally:
            db.close()

    def public_settings(self) -> dict:
        return {
            "enabled": self.settings.xau_paper_enabled,
            "engine_version": PAPER_ENGINE_VERSION,
            "initial_capital": self.settings.xau_paper_initial_capital,
            "risk_pct": self.settings.xau_paper_risk_pct,
            "reward_risk": self.settings.xau_paper_reward_risk,
            "max_leverage": self.settings.xau_paper_max_leverage,
            "max_spread_bps": self.settings.xau_paper_max_spread_bps,
            "max_hold_minutes": self.settings.xau_paper_max_hold_minutes,
            "scan_seconds": self.settings.xau_paper_scan_seconds,
            "timezone": self.settings.xau_paper_timezone,
            "entry_states": ["setup_macro_support", "setup_macro_neutral"],
            "cognition_enabled": self.settings.xau_cognition_enabled,
            "cognition_min_confidence": self.settings.xau_cognition_min_confidence,
            "fast_scan_seconds": self.settings.xau_fast_scan_seconds,
            "reversal_confirmation_cycles": 2,
            "execution_allowed": False,
            "storage": "neon_postgres" if paper_store_is_external() else "local_sqlite_fallback",
            "storage_persistent": paper_store_is_external(),
        }


class XAUPaperTradingScheduler:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.engine = XAUPaperTradingEngine(self.settings)
        self.scheduler = AsyncIOScheduler(timezone=self.settings.xau_paper_timezone)
        self._running = False

    async def _scan(self):
        if self._running:
            return
        self._running = True
        try:
            result = await self.engine.scan()
            account = result.get("account") or {}
            position = result.get("position") or {}
            fusion = result.get("fusion") or {}
            memory = result.get("memory") or {}
            cognition = fusion.get("cognition") or {}
            data_quality = cognition.get("data_quality") or {}
            sensors = data_quality.get("sensors") or {}
            fill = sensors.get("fill_readiness") or {}
            guardian = result.get("position_management") or {}
            logger.info(
                "[XAU paper] week=%s equity=%s position=%s opened=%s closed=%s fusion=%s regime=%s confidence=%s quality=%s quality_issues=%s analysis_ref=%s analysis_age=%s fill_score=%s fill_issues=%s guardian=%s guardian_reason=%s reversal_streak=%s/%s meta=%s memory=%s similar=%s event_risk=%s event_kind=%s event_validation=%s event_policy=%s",
                result.get("week_key"),
                account.get("current_equity"),
                position.get("side") if position else "flat",
                result.get("opened"),
                bool(result.get("closed_trade")),
                fusion.get("state"),
                fusion.get("regime"),
                fusion.get("cognitive_confidence"),
                data_quality.get("score"),
                data_quality.get("issues"),
                sensors.get("analysis_reference_kind"),
                sensors.get("analysis_reference_age_seconds"),
                fill.get("score"),
                fill.get("issues"),
                guardian.get("action"),
                guardian.get("reason"),
                guardian.get("confirmation_streak"),
                guardian.get("confirmation_required"),
                fusion.get("meta_decision"),
                memory.get("source"),
                memory.get("similar_samples"),
                fusion.get("event_risk"),
                fusion.get("event_kind"),
                fusion.get("event_validation"),
                fusion.get("event_policy_version"),
            )
            logger.debug(
                "[XAU cognition] quality=%s hypotheses=%s autopsy_counts=%s calibration=%s",
                (cognition.get("data_quality") or {}).get("score"),
                cognition.get("hypotheses"),
                memory.get("autopsy_counts"),
                {
                    "samples": memory.get("calibration_sample_count"),
                    "brier": memory.get("brier_score"),
                    "ece": memory.get("expected_calibration_error"),
                },
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[XAU paper] scan failed: %s", type(exc).__name__)
        finally:
            self._running = False

    def start(self):
        if not self.settings.xau_paper_enabled:
            logger.info("XAU paper scheduler disabled")
            return
        self.scheduler.add_job(
            self._scan,
            "date",
            run_date=datetime.now(self.scheduler.timezone) + timedelta(seconds=8),
            id="xau_paper_bootstrap",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._scan,
            "interval",
            seconds=max(5, int(self.settings.xau_paper_scan_seconds)),
            id="xau_paper_scan",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "XAU paper scheduler started interval=%ss capital=%.2f",
            self.settings.xau_paper_scan_seconds,
            self.settings.xau_paper_initial_capital,
        )

    def shutdown(self):
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
