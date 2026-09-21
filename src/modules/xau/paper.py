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
from src.modules.xau.service import (
    build_decision_fusion,
    get_macro_context,
    get_xau_snapshot,
)
from src.modules.xau.paper_store import open_xau_paper_session, paper_store_is_external
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

PAPER_ENGINE_VERSION = "0.6.0"


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


def _paper_exit_quote(side: str, spot: dict) -> float | None:
    """Executable-side indicative quote used for simulated exits."""
    if side == "long":
        return _number(spot.get("bid"))
    return _number(spot.get("ask"))


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
        """Retrieve similar historical paper setups, then fall back to global memory."""
        candidate = str(technical.get("candidate") or "none")
        alignment = str(technical.get("alignment") or "mixed")
        micro_direction = str((technical.get("micro") or {}).get("direction") or "neutral")
        frames = technical.get("frames") or {}
        five_direction = str((frames.get("5m") or {}).get("direction") or "neutral")
        fifteen_direction = str((frames.get("15m") or {}).get("direction") or "neutral")
        try:
            macro_bias = max(-1, min(1, int(macro.get("bias", 0))))
        except (TypeError, ValueError):
            macro_bias = 0

        signals = (
            db.query(XAUPaperSignal)
            .filter(
                XAUPaperSignal.accepted == True,  # noqa: E712
                XAUPaperSignal.candidate == candidate,
            )
            .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
            .limit(200)
            .all()
        )

        scored: list[tuple[int, XAUPaperSignal]] = []
        for signal in signals:
            meta = signal.meta or {}
            score = 0
            if str(meta.get("alignment") or "") == alignment:
                score += 3
            saved_micro = meta.get("micro") or {}
            if str(saved_micro.get("direction") or "") == micro_direction:
                score += 2
            saved_frames = meta.get("frames") or {}
            if str((saved_frames.get("5m") or {}).get("direction") or "") == five_direction:
                score += 2
            if str((saved_frames.get("15m") or {}).get("direction") or "") == fifteen_direction:
                score += 2
            try:
                saved_macro = max(-1, min(1, int(meta.get("macro_bias", 0))))
            except (TypeError, ValueError):
                saved_macro = 0
            if saved_macro == macro_bias:
                score += 1
            scored.append((score, signal))

        scored.sort(key=lambda item: item[0], reverse=True)
        similar_signals = [signal for score, signal in scored[:40] if score >= 5]
        setup_keys = [signal.setup_key for signal in similar_signals]

        similar_trades = []
        if setup_keys:
            similar_trades = (
                db.query(XAUPaperTrade)
                .filter(XAUPaperTrade.setup_key.in_(setup_keys))
                .order_by(XAUPaperTrade.closed_at.desc(), XAUPaperTrade.id.desc())
                .limit(40)
                .all()
            )

        if similar_trades:
            metrics = _performance_metrics(similar_trades)
            metrics["source"] = "similar_closed_paper_setups"
            metrics["similar_samples"] = len(similar_trades)
            metrics["candidate"] = candidate
            metrics["alignment"] = alignment
            return metrics

        global_trades = (
            db.query(XAUPaperTrade)
            .order_by(XAUPaperTrade.closed_at.desc(), XAUPaperTrade.id.desc())
            .limit(100)
            .all()
        )
        metrics = _performance_metrics(global_trades)
        metrics["source"] = "global_closed_paper_trades"
        metrics["similar_samples"] = 0
        metrics["candidate"] = candidate
        metrics["alignment"] = alignment
        return metrics

    def _update_account_equity(
        self,
        account: XAUPaperAccount,
        unrealized_pnl: float = 0.0,
    ) -> None:
        equity = float(account.initial_capital) + float(account.realized_pnl) + float(unrealized_pnl)
        account.current_equity = round(equity, 4)
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
                mark = _paper_mark_price(old_position.side, spot)
                source = str(spot.get("source") or old_position.price_source or "last_mark")
                if mark is None:
                    mark = float(old_position.current_price)
                self._close_position(
                    db,
                    active,
                    old_position,
                    float(mark),
                    "weekly_reset",
                    now_utc,
                    {
                        "reset_week": key,
                        "mark_source": source,
                        "fresh_spot": bool(spot) and not bool(spot.get("is_stale")),
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
    ) -> float | None:
        mark = _paper_mark_price(position.side, spot)
        if mark is None:
            return None
        pnl = _pnl(position.side, position.entry_price, mark, position.quantity_oz)
        position.current_price = mark
        position.unrealized_pnl = round(pnl, 4)
        position.mfe_usd = max(float(position.mfe_usd or 0.0), pnl)
        position.mae_usd = min(float(position.mae_usd or 0.0), pnl)
        self._update_account_equity(account, pnl)
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
        now_utc = _utc_naive(now)

        db = open_xau_paper_session()
        try:
            account = self._ensure_week(db, spot, now=now)
            memory = self._memory_snapshot(db, technical, macro)
            fusion = build_decision_fusion(
                technical,
                macro,
                memory=memory,
                min_confidence=float(self.settings.xau_cognition_min_confidence),
            )
            position = self._open_position(db, account.id)
            closed_trade = None

            if position and spot and not bool(spot.get("is_stale")):
                self._mark_position(account, position, spot)
                exit_quote = _paper_exit_quote(position.side, spot)
                if exit_quote is not None:
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

                    if exit_reason is None and position.opened_at:
                        held_minutes = _position_age_minutes(position.opened_at, now_utc)
                        if held_minutes >= float(self.settings.xau_paper_max_hold_minutes):
                            exit_reason = "time_stop"

                    if exit_reason:
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

            if position and spot and not bool(spot.get("is_stale")):
                self._mark_position(account, position, spot)
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
                .filter(XAUPaperSignal.account_id == account.id)
                .order_by(XAUPaperSignal.observed_at.desc(), XAUPaperSignal.id.desc())
                .limit(max(1, min(int(signal_limit), 200)))
                .all()
            )
            return {
                "account": _serialize_account(account),
                "position": _serialize_position(position),
                "trades": [_serialize_trade(item) for item in trades],
                "signals": [_serialize_signal(item) for item in signals],
                "settings": self.public_settings(),
                "performance": _performance_metrics(trades),
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
            logger.info(
                "[XAU paper] week=%s equity=%s position=%s opened=%s closed=%s fusion=%s event_risk=%s event_kind=%s event_validation=%s event_policy=%s",
                result.get("week_key"),
                account.get("current_equity"),
                position.get("side") if position else "flat",
                result.get("opened"),
                bool(result.get("closed_trade")),
                (result.get("fusion") or {}).get("state"),
                (result.get("fusion") or {}).get("event_risk"),
                (result.get("fusion") or {}).get("event_kind"),
                (result.get("fusion") or {}).get("event_validation"),
                (result.get("fusion") or {}).get("event_policy_version"),
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
