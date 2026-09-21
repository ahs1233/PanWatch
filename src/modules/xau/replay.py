"""Research-only walk-forward replay for the XAU cognition stack.

Replay episodes are deliberately isolated from executed paper trades.  Each
decision only sees bars whose candle has already closed at evaluation time.
Future data is used only after the decision to label the forward outcome.

The replay layer never creates broker fills, never marks data execution
eligible, and never feeds its outcomes into trade-calibration metrics.
"""

from __future__ import annotations

import asyncio
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any, Callable

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.paper_store import open_xau_paper_session
from src.platform.marketdata.xau_biquote import BiquoteXAUOHLCProvider
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata.xau_research_provider import YahooGoldResearchProvider
from src.platform.persistence.models import XAUReplayEpisode


_TIMEFRAME_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
}


@dataclass(frozen=True)
class ReplayEpisode:
    replay_key: str
    observed_at: datetime
    outcome_at: datetime
    candidate: str
    regime: str
    confidence: float
    horizon_minutes: int
    entry_price: float
    outcome_price: float
    directional_return_bps: float
    positive: bool
    source: str
    state_vector: dict[str, Any]
    cognition: dict[str, Any]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observed_at"] = self.observed_at.isoformat()
        payload["outcome_at"] = self.outcome_at.isoformat()
        return payload


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _available_at(bar: XAUBar) -> datetime:
    """When a complete candle is knowable to a no-lookahead replay."""
    return _utc(bar.timestamp) + _TIMEFRAME_DURATION[bar.timeframe]


def _sorted_bars(
    bars_by_timeframe: dict[XAUTimeframe, list[XAUBar]],
) -> dict[XAUTimeframe, list[XAUBar]]:
    return {
        timeframe: sorted(
            bars_by_timeframe.get(timeframe) or [],
            key=lambda item: _utc(item.timestamp),
        )
        for timeframe in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15)
    }


def _available_slice(
    bars: list[XAUBar],
    timeframe: XAUTimeframe,
    evaluation_time: datetime,
    *,
    max_bars: int = 300,
) -> list[XAUBar]:
    if not bars:
        return []
    evaluation_time = _utc(evaluation_time)
    availability = [_available_at(bar) for bar in bars]
    end = bisect_right(availability, evaluation_time)
    start = max(0, end - max(30, int(max_bars)))
    return bars[start:end]


def _frame_payload(state, bars: list[XAUBar]) -> dict[str, Any]:
    source = bars[-1].source if bars else "historical_replay"
    return {
        "timeframe": state.timeframe.value,
        "source": source,
        "close": state.close,
        "ema_fast": state.ema_fast,
        "ema_slow": state.ema_slow,
        "rsi14": state.rsi14,
        "atr14": state.atr14,
        "atr_pct": state.atr_pct,
        "breakout": state.breakout,
        "direction": state.direction,
        "recent_swing_high": state.recent_swing_high,
        "recent_swing_low": state.recent_swing_low,
        "observed_at": _utc(state.observed_at).isoformat(),
    }


def _micro_payload(m1: list[XAUBar], evaluation_time: datetime) -> dict[str, Any]:
    latest = m1[-1]
    price = float(latest.close)

    def return_pct(lookback: int) -> float:
        if len(m1) <= lookback:
            return 0.0
        previous = float(m1[-(lookback + 1)].close)
        return ((price - previous) / previous) * 100.0 if previous else 0.0

    ret10 = return_pct(10)
    ret30 = return_pct(30)
    direction = (
        "bullish"
        if ret10 > 0.02
        else "bearish"
        if ret10 < -0.02
        else "neutral"
    )
    return {
        "status": "ready",
        "direction": direction,
        "price": price,
        "return_10m_pct": ret10,
        "return_30m_pct": ret30,
        "age_seconds": 0.0,
        "is_stale": False,
        "source": "historical_replay:1m",
        "observed_at": _utc(evaluation_time).isoformat(),
        "last_point_at": _utc(evaluation_time).isoformat(),
    }


def build_replay_technical_state(
    bars_by_timeframe: dict[XAUTimeframe, list[XAUBar]],
    evaluation_time: datetime,
    *,
    macro_bias: int = 0,
) -> dict[str, Any]:
    """Build a live-shaped technical snapshot using only closed historical bars."""
    evaluation_time = _utc(evaluation_time)
    available = {
        timeframe: _available_slice(
            bars_by_timeframe.get(timeframe) or [],
            timeframe,
            evaluation_time,
        )
        for timeframe in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15)
    }
    assessment = XAUIntradayEngine(require_execution_data=False).analyze(
        available,
        event_risk=False,
        macro_bias=macro_bias,
        now=evaluation_time,
    )

    frames = {
        name: _frame_payload(
            state,
            available.get(state.timeframe) or [],
        )
        for name, state in assessment.frame_states.items()
    }
    m1 = available[XAUTimeframe.M1]
    if not m1:
        return {
            "status": "blocked",
            "candidate": "none",
            "blocked": True,
            "block_reasons": ["insufficient_1m_bars"],
            "frames": frames,
            "observed_at": evaluation_time.isoformat(),
            "technical_mode": "historical_replay",
            "research_only": True,
            "execution_status": "LOCKED_REPLAY_RESEARCH_ONLY",
        }

    micro = _micro_payload(m1, evaluation_time)
    directions = [
        str(micro.get("direction") or "neutral"),
        str((frames.get("5m") or {}).get("direction") or "neutral"),
        str((frames.get("15m") or {}).get("direction") or "neutral"),
    ]
    bullish = sum(1 for value in directions if value == "bullish")
    bearish = sum(1 for value in directions if value == "bearish")
    alignment = "bullish" if bullish >= 2 else "bearish" if bearish >= 2 else "mixed"

    price = float(m1[-1].close)
    candidate = assessment.candidate if not assessment.blocked else "none"
    reference = {
        "price": price,
        "source": "historical_replay:1m",
        "observed_at": evaluation_time.isoformat(),
        "age_seconds": 0.0,
        "is_stale": False,
        "kind": "historical_replay",
        "execution_eligible": False,
    }
    spot = {
        "price": price,
        "bid": None,
        "ask": None,
        "spread": None,
        "spread_bps": None,
        "observed_at": evaluation_time.isoformat(),
        "age_seconds": 0.0,
        "source": "historical_replay:mid",
        "is_stale": False,
        "fill_state": "unavailable",
        "provider_health": [],
        "indicative": True,
        "execution_eligible": False,
    }

    return {
        "instrument": "XAUUSD",
        "research_only": True,
        "execution_feed_connected": False,
        "execution_status": "LOCKED_REPLAY_RESEARCH_ONLY",
        "price": price,
        "indicative_spot": spot,
        "analysis_reference": reference,
        "spot_consensus": {
            "usable_count": 1,
            "primary_delta_bps": 0.0,
            "reference_median": price,
            "research_replay": True,
        },
        "micro": micro,
        "technical_mode": "historical_replay",
        "spot_minus_proxy": 0.0,
        "spot_minus_proxy_bps": 0.0,
        "observed_at": evaluation_time.isoformat(),
        "status": assessment.status,
        "candidate": candidate,
        "blocked": assessment.blocked,
        "block_reasons": list(assessment.block_reasons),
        "raw_proxy_block_reasons": list(assessment.block_reasons),
        "warnings": [],
        "alignment": alignment,
        "atr_reference": assessment.atr_reference,
        "swing_high_reference": assessment.swing_high_reference,
        "swing_low_reference": assessment.swing_low_reference,
        "frames": frames,
    }


def _future_close(
    m1: list[XAUBar],
    target_time: datetime,
    *,
    tolerance_minutes: int = 2,
) -> tuple[float, datetime] | None:
    """Find the first fully closed 1m bar at/after the target horizon."""
    if not m1:
        return None
    target = _utc(target_time)
    availability = [_available_at(bar) for bar in m1]
    index = bisect_left(availability, target)
    if index >= len(m1):
        return None
    outcome_at = availability[index]
    if outcome_at - target > timedelta(minutes=max(0, int(tolerance_minutes))):
        return None
    return float(m1[index].close), outcome_at


def walk_forward_replay(
    bars_by_timeframe: dict[XAUTimeframe, list[XAUBar]],
    *,
    horizon_minutes: int = 60,
    step_minutes: int = 5,
    macro_provider: Callable[[datetime], dict[str, Any]] | None = None,
    source: str = "historical_replay",
    min_confidence: float = 0.58,
) -> list[ReplayEpisode]:
    """Generate no-lookahead directional research episodes.

    Input timestamps are treated as candle-open timestamps. A candle becomes
    visible only after its full timeframe duration has elapsed.
    """
    horizon_minutes = max(1, int(horizon_minutes))
    step_minutes = max(1, int(step_minutes))
    bars = _sorted_bars(bars_by_timeframe)
    m1 = bars[XAUTimeframe.M1]
    if not m1:
        return []

    episodes: list[ReplayEpisode] = []
    last_eval: datetime | None = None

    for bar in m1:
        evaluation_time = _available_at(bar)
        if last_eval is not None and evaluation_time - last_eval < timedelta(minutes=step_minutes):
            continue
        last_eval = evaluation_time

        macro = (
            dict(macro_provider(evaluation_time) or {})
            if macro_provider is not None
            else {}
        )
        macro.setdefault("bias", 0)
        macro.setdefault("confidence", 0.0)
        macro.setdefault("event_risk", False)

        technical = build_replay_technical_state(
            bars,
            evaluation_time,
            macro_bias=int(macro.get("bias", 0) or 0),
        )
        candidate = str(technical.get("candidate") or "none")
        if technical.get("blocked") or candidate not in {"long_setup", "short_setup"}:
            continue

        cognition = build_cognitive_state(
            technical,
            macro,
            memory=None,
            min_confidence=min_confidence,
        )
        future = _future_close(
            m1,
            evaluation_time + timedelta(minutes=horizon_minutes),
        )
        if future is None:
            continue
        outcome_price, outcome_at = future

        entry_price = float(technical["analysis_reference"]["price"])
        side = 1.0 if candidate == "long_setup" else -1.0
        directional_return_bps = (
            ((outcome_price - entry_price) / entry_price) * 10_000.0 * side
        )
        regime = str((cognition.get("regime") or {}).get("label") or "")
        confidence = float(
            (cognition.get("confidence") or {}).get("calibrated_confidence")
            or 0.0
        )
        key_material = (
            f"{source}|{evaluation_time.isoformat()}|{candidate}|"
            f"{horizon_minutes}|{entry_price:.6f}"
        )
        replay_key = sha1(key_material.encode("utf-8")).hexdigest()

        episodes.append(
            ReplayEpisode(
                replay_key=replay_key,
                observed_at=evaluation_time,
                outcome_at=outcome_at,
                candidate=candidate,
                regime=regime,
                confidence=round(confidence, 6),
                horizon_minutes=horizon_minutes,
                entry_price=round(entry_price, 6),
                outcome_price=round(outcome_price, 6),
                directional_return_bps=round(directional_return_bps, 4),
                positive=directional_return_bps > 0,
                source=source,
                state_vector=dict(cognition.get("market_state") or {}),
                cognition=cognition,
                meta={
                    "research_only": True,
                    "lookahead_protected": True,
                    "step_minutes": step_minutes,
                    "macro_bias": macro.get("bias"),
                    "macro_confidence": macro.get("confidence"),
                    "event_risk": bool(macro.get("event_risk")),
                },
            )
        )

    return episodes


def persist_replay_episodes(db, episodes: list[ReplayEpisode]) -> int:
    """Idempotently persist research replay episodes in the XAU durable store."""
    if not episodes:
        return 0
    keys = [episode.replay_key for episode in episodes]
    existing = {
        row[0]
        for row in (
            db.query(XAUReplayEpisode.replay_key)
            .filter(XAUReplayEpisode.replay_key.in_(keys))
            .all()
        )
    }
    added = 0
    for episode in episodes:
        if episode.replay_key in existing:
            continue
        db.add(
            XAUReplayEpisode(
                replay_key=episode.replay_key,
                candidate=episode.candidate,
                regime=episode.regime,
                confidence=episode.confidence,
                horizon_minutes=episode.horizon_minutes,
                entry_price=episode.entry_price,
                outcome_price=episode.outcome_price,
                directional_return_bps=episode.directional_return_bps,
                positive=episode.positive,
                source=episode.source,
                observed_at=episode.observed_at.replace(tzinfo=None),
                outcome_at=episode.outcome_at.replace(tzinfo=None),
                state_vector=episode.state_vector,
                cognition=episode.cognition,
                meta=episode.meta,
            )
        )
        added += 1
    if added:
        db.flush()
    return added


async def _fetch_default_replay_history(
    *,
    limit: int = 1000,
) -> tuple[dict[XAUTimeframe, list[XAUBar]], str]:
    """Fetch one internally consistent replay dataset.

    Biquote spot/MT5 structure is preferred for all timeframes. If any required
    timeframe fails or is too short, replay falls back to Yahoo GC=F for all
    frames rather than mixing instruments inside one historical episode.
    """
    limit = max(60, min(int(limit), 1000))
    biquote = BiquoteXAUOHLCProvider()

    async def biquote_one(timeframe: XAUTimeframe):
        return await asyncio.to_thread(
            biquote.bars,
            timeframe,
            limit=limit,
        )

    try:
        results = await asyncio.gather(
            *(biquote_one(tf) for tf in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15))
        )
        bars = {
            XAUTimeframe.M1: results[0],
            XAUTimeframe.M5: results[1],
            XAUTimeframe.M15: results[2],
        }
        if all(len(rows) >= 30 for rows in bars.values()):
            return bars, "biquote.io:MT5-ohlc"
    except Exception:
        pass

    yahoo = YahooGoldResearchProvider()

    async def yahoo_one(timeframe: XAUTimeframe):
        return await asyncio.to_thread(yahoo.bars, timeframe)

    results = await asyncio.gather(
        *(yahoo_one(tf) for tf in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15))
    )
    bars = {
        XAUTimeframe.M1: results[0],
        XAUTimeframe.M5: results[1],
        XAUTimeframe.M15: results[2],
    }
    if not all(len(rows) >= 30 for rows in bars.values()):
        raise RuntimeError("insufficient historical XAU replay bars")
    return bars, "yfinance:GC=F"


async def refresh_replay_memory(
    *,
    horizon_minutes: int = 60,
    step_minutes: int = 5,
    limit: int = 1000,
) -> dict[str, Any]:
    """Fetch historical bars, run no-lookahead replay and persist new episodes."""
    bars, source = await _fetch_default_replay_history(limit=limit)
    replay_source = f"{source}:walk-forward"
    episodes = walk_forward_replay(
        bars,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        source=replay_source,
    )

    db = open_xau_paper_session()
    try:
        added = persist_replay_episodes(db, episodes)
        db.commit()
        total = (
            db.query(XAUReplayEpisode)
            .filter(XAUReplayEpisode.source == replay_source)
            .count()
        )
    finally:
        db.close()

    observed_times = [episode.observed_at for episode in episodes]
    return {
        "status": "ok",
        "source": source,
        "replay_source": replay_source,
        "bars": {
            timeframe.value: len(rows)
            for timeframe, rows in bars.items()
        },
        "generated_episodes": len(episodes),
        "added_episodes": added,
        "stored_episodes_for_source": total,
        "horizon_minutes": int(horizon_minutes),
        "step_minutes": int(step_minutes),
        "first_observed_at": (
            min(observed_times).isoformat() if observed_times else None
        ),
        "last_observed_at": (
            max(observed_times).isoformat() if observed_times else None
        ),
        "research_only": True,
        "execution_allowed": False,
        "lookahead_protected": True,
    }
