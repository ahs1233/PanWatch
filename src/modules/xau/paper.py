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
from src.modules.xau.service import (
    build_decision_fusion,
    get_macro_context,
    get_xau_snapshot,
)
from src.modules.xau.paper_store import open_xau_paper_session
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)


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


def _paper_entry_price(side: str, spot: dict) -> float | None:
    if side == "long":
        return _number(spot.get("ask")) or _number(spot.get("price"))
    return _number(spot.get("bid")) or _number(spot.get("price"))


def _paper_mark_price(side: str, spot: dict) -> float | None:
    if side == "long":
        return _number(spot.get("bid")) or _number(spot.get("price"))
    return _number(spot.get("ask")) or _number(spot.get("price"))


def _pnl(side: str, entry: float, mark: float, quantity: float) -> float:
    if side == "long":
        return (mark - entry) * quantity
    return (entry - mark) * quantity


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
            meta=meta or {},
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
            f"{fusion.get('state')}:"
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
                "technical_mode": technical.get("technical_mode"),
                "alignment": technical.get("alignment"),
                "macro_bias": macro.get("bias"),
                "macro_confidence": macro.get("confidence"),
                "fusion_reasons": fusion.get("reasons") or [],
                "spot_source": spot.get("source"),
                "spot_stale": spot.get("is_stale"),
                "execution_allowed": False,
            },
        )
        db.add(signal)
        db.flush()
        return signal

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
        if candidate not in {"long_setup", "short_setup"}:
            return None

        state = str(fusion.get("state") or "")
        accepted_state = state in {"setup_macro_support", "setup_macro_neutral"}
        rejection_reason = ""
        if bool(spot.get("is_stale")):
            accepted_state = False
            rejection_reason = "indicative_spot_stale"
        elif not accepted_state:
            rejection_reason = state or "fusion_not_eligible"

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
        if not signal or not accepted_state:
            return None

        if self._open_position(db, account.id):
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
            signal.rejection_reason = "paper_entry_price_unavailable"
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
        db.add(position)
        db.flush()

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

    async def scan(self, now: datetime | None = None) -> dict:
        if not self.settings.xau_paper_enabled:
            return {"status": "disabled", "execution_allowed": False}

        technical = await get_xau_snapshot(force=False)
        macro = await get_macro_context(force=False)
        fusion = build_decision_fusion(technical, macro)
        spot = technical.get("indicative_spot") or {}
        now_utc = _utc_naive(now)

        db = open_xau_paper_session()
        try:
            account = self._ensure_week(db, spot, now=now)
            position = self._open_position(db, account.id)
            closed_trade = None

            if position and spot and not bool(spot.get("is_stale")):
                mark = self._mark_position(account, position, spot)
                if mark is not None:
                    exit_reason = None
                    if position.side == "long":
                        if mark <= position.stop_loss:
                            exit_reason = "stop_loss"
                        elif mark >= position.target_price:
                            exit_reason = "target_price"
                    else:
                        if mark >= position.stop_loss:
                            exit_reason = "stop_loss"
                        elif mark <= position.target_price:
                            exit_reason = "target_price"

                    if exit_reason:
                        closed_trade = self._close_position(
                            db,
                            account,
                            position,
                            mark,
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
                "execution_allowed": False,
            }
        finally:
            db.close()

    def public_settings(self) -> dict:
        return {
            "enabled": self.settings.xau_paper_enabled,
            "initial_capital": self.settings.xau_paper_initial_capital,
            "risk_pct": self.settings.xau_paper_risk_pct,
            "reward_risk": self.settings.xau_paper_reward_risk,
            "max_leverage": self.settings.xau_paper_max_leverage,
            "scan_seconds": self.settings.xau_paper_scan_seconds,
            "timezone": self.settings.xau_paper_timezone,
            "entry_states": ["setup_macro_support", "setup_macro_neutral"],
            "execution_allowed": False,
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
                "[XAU paper] week=%s equity=%s position=%s opened=%s closed=%s",
                result.get("week_key"),
                account.get("current_equity"),
                position.get("side") if position else "flat",
                result.get("opened"),
                bool(result.get("closed_trade")),
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
            seconds=max(30, int(self.settings.xau_paper_scan_seconds)),
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
