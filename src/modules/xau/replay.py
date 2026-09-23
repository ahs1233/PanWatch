"""Research-only walk-forward replay for the XAU cognition stack.

Replay episodes are deliberately isolated from executed paper trades.  Each
decision only sees bars whose candle has already closed at evaluation time.
Future data is used only after the decision to label the forward outcome.

The replay layer never creates broker fills, never marks data execution
eligible, and never feeds its outcomes into trade-calibration metrics.
"""

from __future__ import annotations

import asyncio
import logging
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha1
from typing import Any, Callable

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
from src.modules.xau.market_context import build_market_context
from src.modules.xau.service import build_decision_fusion
from src.modules.xau.paper_store import (
    open_xau_paper_session,
    open_xau_replay_session,
    paper_store_is_external,
    replay_store_is_external,
)
from src.platform.marketdata.xau_biquote import BiquoteXAUOHLCProvider
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata.xau_research_provider import YahooGoldResearchProvider
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperSignal,
    XAUReplayEpisode,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)


_TIMEFRAME_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.M30: timedelta(minutes=30),
    XAUTimeframe.H1: timedelta(hours=1),
    XAUTimeframe.H4: timedelta(hours=4),
    XAUTimeframe.D1: timedelta(days=1),
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
        for timeframe in (
            XAUTimeframe.M1,
            XAUTimeframe.M5,
            XAUTimeframe.M15,
            XAUTimeframe.H1,
            XAUTimeframe.H4,
            XAUTimeframe.D1,
        )
    }


def _available_slice(
    bars: list[XAUBar],
    timeframe: XAUTimeframe,
    evaluation_time: datetime,
    *,
    max_bars: int = 300,
    availability: list[datetime] | None = None,
) -> list[XAUBar]:
    if not bars:
        return []
    evaluation_time = _utc(evaluation_time)
    availability = availability or [_available_at(bar) for bar in bars]
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
    availability_by_timeframe: dict[XAUTimeframe, list[datetime]] | None = None,
    market_context_builder: Callable[
        [list[XAUBar], list[XAUBar], list[XAUBar]],
        dict[str, Any],
    ] | None = None,
) -> dict[str, Any]:
    """Build a live-shaped technical snapshot using only closed historical bars."""
    evaluation_time = _utc(evaluation_time)
    available = {
        timeframe: _available_slice(
            bars_by_timeframe.get(timeframe) or [],
            timeframe,
            evaluation_time,
            max_bars=1000 if timeframe in {XAUTimeframe.H1, XAUTimeframe.H4, XAUTimeframe.D1} else 300,
            availability=(availability_by_timeframe or {}).get(timeframe),
        )
        for timeframe in (
            XAUTimeframe.M1,
            XAUTimeframe.M5,
            XAUTimeframe.M15,
            XAUTimeframe.H1,
            XAUTimeframe.H4,
            XAUTimeframe.D1,
        )
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

    h1 = available.get(XAUTimeframe.H1) or []
    h4 = available.get(XAUTimeframe.H4) or []
    daily = available.get(XAUTimeframe.D1) or []
    context_builder = market_context_builder or build_market_context
    market_context = (
        context_builder(h1, h4, daily)
        if h1 or h4 or daily
        else {}
    )

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
        "market_context": market_context,
        "market_context_error": None if market_context else "historical_htf_unavailable",
        "xaut_order_flow": None,
        "xaut_order_flow_error": "historical_xaut_microstructure_unavailable",
    }


def _future_range_outcomes(
    m1: list[XAUBar],
    evaluation_time: datetime,
    entry_price: float,
    *,
    horizon_minutes: int,
    distances: tuple[int, ...] = (10, 20, 30),
    availability: list[datetime] | None = None,
) -> dict[str, Any]:
    """Measure first-touch +/- USD outcomes using future closed M1 OHLC only."""
    start = _utc(evaluation_time)
    end = start + timedelta(minutes=max(1, int(horizon_minutes)))
    availability = availability or [_available_at(bar) for bar in m1]
    start_index = bisect_right(availability, start)
    end_index = bisect_right(availability, end)
    future = m1[start_index:end_index]
    out: dict[str, Any] = {
        "horizon_minutes": int(horizon_minutes),
        "bar_count": len(future),
        "lookahead_used_for_label_only": True,
        "levels": {},
    }
    if not future:
        return out

    max_up = max(float(bar.high) - entry_price for bar in future)
    max_down = max(entry_price - float(bar.low) for bar in future)
    out["max_up_usd"] = round(max_up, 4)
    out["max_down_usd"] = round(max_down, 4)

    for distance in distances:
        up_level = entry_price + float(distance)
        down_level = entry_price - float(distance)
        first_hit = "none"
        first_hit_at = None
        up_hit = False
        down_hit = False
        for bar in future:
            hit_up = float(bar.high) >= up_level
            hit_down = float(bar.low) <= down_level
            up_hit = up_hit or hit_up
            down_hit = down_hit or hit_down
            if first_hit == "none" and (hit_up or hit_down):
                first_hit_at = _available_at(bar).isoformat()
                if hit_up and hit_down:
                    first_hit = "ambiguous_same_bar"
                else:
                    first_hit = "up" if hit_up else "down"
        out["levels"][f"pm{distance}"] = {
            "distance_usd": distance,
            "up_level": round(up_level, 4),
            "down_level": round(down_level, 4),
            "up_hit": up_hit,
            "down_hit": down_hit,
            "first_hit": first_hit,
            "first_hit_at": first_hit_at,
        }
    return out


def _future_close(
    m1: list[XAUBar],
    target_time: datetime,
    *,
    tolerance_minutes: int = 2,
    availability: list[datetime] | None = None,
) -> tuple[float, datetime] | None:
    """Find the first fully closed 1m bar at/after the target horizon."""
    if not m1:
        return None
    target = _utc(target_time)
    availability = availability or [_available_at(bar) for bar in m1]
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
    availability_by_timeframe = {
        timeframe: [_available_at(bar) for bar in rows]
        for timeframe, rows in bars.items()
    }
    m1 = bars[XAUTimeframe.M1]
    m1_availability = availability_by_timeframe[XAUTimeframe.M1]
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
        macro.setdefault("observed_at", evaluation_time.isoformat())
        if macro_provider is not None:
            # A supplied historical provider is responsible for point-in-time
            # evidence. Missing readiness fields remain explicit, but defaults
            # make simple deterministic providers usable in research replay.
            macro.setdefault("calendar_ok", True)
            macro.setdefault("search_ok", True)
            macro.setdefault("synthesis_ok", True)
            macro.setdefault("cache_stale", False)
            macro.setdefault("refresh_pending", False)
        else:
            macro.setdefault("calendar_ok", False)
            macro.setdefault("search_ok", False)
            macro.setdefault("synthesis_ok", False)
            macro.setdefault("cache_stale", False)
            macro.setdefault("refresh_pending", False)
            macro.setdefault("historical_gap", "macro_provider_not_supplied")

        technical = build_replay_technical_state(
            bars,
            evaluation_time,
            macro_bias=int(macro.get("bias", 0) or 0),
            availability_by_timeframe=availability_by_timeframe,
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
        # Replay the same Gen1 decision stack wherever historical inputs exist.
        # Historical XAUT raw book/trade tape is intentionally absent, so the
        # evidence layer runs with require_xaut=False and records that gap.
        fusion = build_decision_fusion(
            technical,
            macro,
            memory=None,
            min_confidence=min_confidence,
            as_of=evaluation_time,
        )
        evidence = build_gen1_evidence_fusion(
            technical,
            macro,
            fusion,
            memory=None,
            require_xaut=False,
        )
        gen1_decision = str(evidence.get("decision") or "WAIT")
        gen1_confidence = evidence.get("decision_confidence")
        sensor_gaps = ["historical_xaut_microstructure_unavailable"]
        if not technical.get("market_context"):
            sensor_gaps.append("historical_htf_context_unavailable")
        if macro_provider is None:
            sensor_gaps.append("historical_macro_unavailable")
        future = _future_close(
            m1,
            evaluation_time + timedelta(minutes=horizon_minutes),
            availability=m1_availability,
        )
        if future is None:
            continue
        outcome_price, outcome_at = future

        entry_price = float(technical["analysis_reference"]["price"])
        range_outcomes = _future_range_outcomes(
            m1,
            evaluation_time,
            entry_price,
            horizon_minutes=horizon_minutes,
            availability=m1_availability,
        )
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
                    "gen1_decision": gen1_decision,
                    "gen1_decision_confidence": gen1_confidence,
                    "gen1_fusion_state": fusion.get("state"),
                    "evidence_fusion": evidence,
                    "range_outcomes": range_outcomes,
                    "sensor_gaps": sensor_gaps,
                    "replay_scope": "full_gen1_except_historical_xaut_microstructure",
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


def persist_replay_episodes_in_paper_store(db, episodes: list[ReplayEpisode]) -> int:
    """Durable no-DDL fallback using isolated replay_* rows in paper signals.

    These rows are never accepted trades and use candidate names that normal
    paper/shadow queries do not match.
    """
    if not episodes:
        return 0

    account = (
        db.query(XAUPaperAccount)
        .filter(XAUPaperAccount.status == "active")
        .order_by(XAUPaperAccount.id.desc())
        .first()
    )
    if account is None:
        account = (
            db.query(XAUPaperAccount)
            .order_by(XAUPaperAccount.id.desc())
            .first()
        )
    if account is None:
        return 0

    setup_keys = [f"replay:{episode.replay_key}" for episode in episodes]
    existing = {
        row[0]
        for row in (
            db.query(XAUPaperSignal.setup_key)
            .filter(XAUPaperSignal.setup_key.in_(setup_keys))
            .all()
        )
    }

    added = 0
    for episode in episodes:
        setup_key = f"replay:{episode.replay_key}"
        if setup_key in existing:
            continue
        replay_candidate = f"replay_{episode.candidate}"
        db.add(
            XAUPaperSignal(
                account_id=account.id,
                setup_key=setup_key,
                candidate=replay_candidate,
                fusion_state="historical_replay",
                macro_relation="research_only",
                event_risk=False,
                price=episode.entry_price,
                accepted=False,
                rejection_reason="historical_replay",
                observed_at=episode.observed_at.replace(tzinfo=None),
                meta={
                    "research_only": True,
                    "lookahead_protected": True,
                    "replay_episode": episode.to_dict(),
                    "state_vector": episode.state_vector,
                    "cognition": episode.cognition,
                    "directional_return_bps": episode.directional_return_bps,
                    "outcome_at": episode.outcome_at.isoformat(),
                    "horizon_minutes": episode.horizon_minutes,
                    "source": episode.source,
                },
            )
        )
        added += 1

    if added:
        db.flush()
    return added


async def _fetch_default_replay_history(
    *,
    limit: int = 1000,
    lookback_days: int = 0,
) -> tuple[dict[XAUTimeframe, list[XAUBar]], str]:
    """Fetch one internally consistent replay dataset.

    Biquote spot/MT5 structure is preferred for all timeframes. If any required
    timeframe fails or is too short, replay falls back to Yahoo GC=F for all
    frames rather than mixing instruments inside one historical episode.
    """
    limit = max(60, min(int(limit), 1000))
    lookback_days = max(0, min(int(lookback_days), 30))
    biquote = BiquoteXAUOHLCProvider()

    if lookback_days > 0:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)

        async def biquote_range_one(timeframe: XAUTimeframe):
            return await asyncio.to_thread(
                biquote.bars_range,
                timeframe,
                start=start,
                end=end,
            )

        try:
            results = await asyncio.gather(
                *(biquote_range_one(tf) for tf in (
                    XAUTimeframe.M1,
                    XAUTimeframe.M5,
                    XAUTimeframe.M15,
                ))
            )
            deep_bars = {
                XAUTimeframe.M1: results[0],
                XAUTimeframe.M5: results[1],
                XAUTimeframe.M15: results[2],
            }
            minimums = {
                XAUTimeframe.M1: 300,
                XAUTimeframe.M5: 60,
                XAUTimeframe.M15: 30,
            }
            if all(
                len(deep_bars[tf]) >= minimums[tf]
                for tf in minimums
            ):
                return deep_bars, "biquote.io:MT5-ohlc-range"
        except Exception as exc:  # noqa: BLE001 - fail soft to recent history
            logger.warning(
                "[XAU replay] deep MT5 history unavailable type=%s; using recent fallback",
                type(exc).__name__,
            )

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


async def _attach_optional_htf_history(
    bars: dict[XAUTimeframe, list[XAUBar]],
    *,
    lookback_days: int,
    source: str,
) -> dict[XAUTimeframe, list[XAUBar]]:
    """Best-effort HTF context from the same source family as replay core."""
    use_biquote = str(source).startswith("biquote.io:")
    provider = BiquoteXAUOHLCProvider() if use_biquote else YahooGoldResearchProvider()
    end = datetime.now(timezone.utc)
    days = max(5, int(lookback_days or 30))
    start = end - timedelta(days=min(365, max(days, 30)))

    async def one(tf: XAUTimeframe):
        try:
            if use_biquote:
                if lookback_days > 0:
                    return await asyncio.to_thread(
                        provider.bars_range,
                        tf,
                        start=start,
                        end=end,
                    )
                return await asyncio.to_thread(provider.bars, tf, limit=1000)
            # Yahoo/GC=F must remain GC=F. Its provider supports H1/D1;
            # H4 is derived from H1 by build_market_context.
            if tf == XAUTimeframe.H4:
                return []
            return await asyncio.to_thread(provider.bars, tf)
        except Exception:
            return []

    h1, h4, d1 = await asyncio.gather(
        one(XAUTimeframe.H1),
        one(XAUTimeframe.H4),
        one(XAUTimeframe.D1),
    )
    enriched = dict(bars)
    enriched[XAUTimeframe.H1] = h1
    enriched[XAUTimeframe.H4] = h4
    enriched[XAUTimeframe.D1] = d1
    return enriched


async def refresh_replay_memory(
    *,
    horizon_minutes: int = 60,
    step_minutes: int = 5,
    limit: int = 1000,
    lookback_days: int = 0,
) -> dict[str, Any]:
    """Fetch historical bars, run no-lookahead replay and persist new episodes."""
    bars, source = await _fetch_default_replay_history(
        limit=limit,
        lookback_days=lookback_days,
    )
    bars = await _attach_optional_htf_history(
        bars,
        lookback_days=lookback_days,
        source=source,
    )
    return await asyncio.to_thread(
        _replay_and_persist, bars, source,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        lookback_days=lookback_days,
    )


def _replay_and_persist(
    bars, source: str, *, horizon_minutes: int, step_minutes: int, lookback_days: int,
) -> dict[str, Any]:
    """Run replay and own its entire database session outside the event loop."""
    replay_source = f"{source}:walk-forward"
    episodes = walk_forward_replay(
        bars,
        horizon_minutes=horizon_minutes,
        step_minutes=step_minutes,
        source=replay_source,
    )

    if replay_store_is_external():
        db = open_xau_replay_session()
        try:
            added = persist_replay_episodes(db, episodes)
            db.commit()
            total = (
                db.query(XAUReplayEpisode)
                .filter(XAUReplayEpisode.source == replay_source)
                .count()
            )
            storage_mode = "external_replay_table"
        finally:
            db.close()
    else:
        db = open_xau_paper_session()
        try:
            added = persist_replay_episodes_in_paper_store(db, episodes)
            db.commit()
            total = (
                db.query(XAUPaperSignal)
                .filter(
                    XAUPaperSignal.rejection_reason == "historical_replay",
                    XAUPaperSignal.candidate.in_(
                        ("replay_long_setup", "replay_short_setup")
                    ),
                )
                .count()
            )
            storage_mode = "external_paper_signal_compat"
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
        "lookback_days": int(lookback_days),
        "first_observed_at": (
            min(observed_times).isoformat() if observed_times else None
        ),
        "last_observed_at": (
            max(observed_times).isoformat() if observed_times else None
        ),
        "research_only": True,
        "execution_allowed": False,
        "lookahead_protected": True,
        "durable_external_store": (
            replay_store_is_external()
            or (
                storage_mode == "external_paper_signal_compat"
                and paper_store_is_external()
            )
        ),
        "storage_mode": storage_mode,
    }


class XAUReplayScheduler:
    """Low-frequency replay refresh isolated from the live XAU hot path."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.scheduler = AsyncIOScheduler(timezone=self.settings.xau_paper_timezone)

    async def _refresh(self) -> None:
        try:
            result = await refresh_replay_memory(
                horizon_minutes=self.settings.xau_replay_horizon_minutes,
                step_minutes=self.settings.xau_replay_step_minutes,
                limit=self.settings.xau_replay_bar_limit,
                lookback_days=self.settings.xau_replay_lookback_days,
            )
            logger.info(
                "[XAU replay] source=%s generated=%s added=%s stored=%s storage=%s durable=%s lookback_days=%s range=%s..%s research_only=true",
                result.get("source"),
                result.get("generated_episodes"),
                result.get("added_episodes"),
                result.get("stored_episodes_for_source"),
                result.get("storage_mode"),
                result.get("durable_external_store"),
                result.get("lookback_days"),
                result.get("first_observed_at"),
                result.get("last_observed_at"),
            )
        except Exception as exc:  # noqa: BLE001 - replay must never block live runtime
            logger.warning(
                "[XAU replay] refresh failed type=%s",
                type(exc).__name__,
            )

    def start(self) -> None:
        if not self.settings.xau_replay_enabled:
            logger.info("[XAU replay] scheduler disabled")
            return
        interval = max(60, int(self.settings.xau_replay_interval_minutes))
        first_run = datetime.now(timezone.utc) + timedelta(seconds=90)
        self.scheduler.add_job(
            self._refresh,
            trigger="interval",
            minutes=interval,
            next_run_time=first_run,
            id="xau-replay-refresh",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "[XAU replay] scheduler started interval_minutes=%s first_run_delay_seconds=90",
            interval,
        )

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
