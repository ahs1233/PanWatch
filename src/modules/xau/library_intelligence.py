"""Third-party XAU market-intelligence adapters.

Primary runtime signal: pyvsmc.
Independent oracles: smartmoneyconcepts and smc-mcp.
Indicator parity: pandas-ta-classic.
Volume-profile oracle: MarketProfile.
Historical research/backtest harness: vectorbt.

Every adapter is fail-soft: a library failure is surfaced as status/error and
never silently converted into a bullish/bearish signal.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from src.platform.marketdata.xau_models import XAUBar


def _frame(rows: list[XAUBar]):
    import pandas as pd

    data = {
        "open": [float(r.open) for r in rows],
        "high": [float(r.high) for r in rows],
        "low": [float(r.low) for r in rows],
        "close": [float(r.close) for r in rows],
        "volume": [float(r.volume or 0.0) for r in rows],
    }
    return pd.DataFrame(data, index=pd.DatetimeIndex([r.timestamp for r in rows]))


def _native(value: Any) -> Any:
    if is_dataclass(value):
        return {k: _native(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _last_signal(values: Any) -> tuple[int | None, float | int | bool | None]:
    try:
        seq = list(values)
    except Exception:
        return None, None
    for idx in range(len(seq) - 1, -1, -1):
        value = _native(seq[idx])
        try:
            active = bool(value) and float(value) != 0.0
        except Exception:
            active = bool(value)
        if active:
            return idx, value
    return None, None


def pandas_ta_snapshot(rows: list[XAUBar]) -> dict[str, Any]:
    if len(rows) < 20:
        return {"status": "insufficient_data", "library": "pandas-ta-classic"}
    try:
        import pandas_ta_classic as ta
    except Exception as exc:
        return {"status": "unavailable", "library": "pandas-ta-classic", "error": type(exc).__name__}

    df = _frame(rows)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    out: dict[str, Any] = {"status": "ok", "library": "pandas-ta-classic"}
    try:
        for period in (9, 21, 50, 200, 1000):
            series = ta.ema(close, length=period)
            value = series.iloc[-1] if series is not None and len(series) else None
            out[f"ema{period}"] = None if value is None or value != value else round(float(value), 6)
        rsi = ta.rsi(close, length=14)
        atr = ta.atr(high, low, close, length=14)
        cmf = ta.cmf(high, low, close, volume, length=20)
        obv = ta.obv(close, volume)
        adx = ta.adx(high, low, close, length=14)
        out["rsi14"] = round(float(rsi.iloc[-1]), 4) if rsi is not None and rsi.iloc[-1] == rsi.iloc[-1] else None
        out["atr14"] = round(float(atr.iloc[-1]), 6) if atr is not None and atr.iloc[-1] == atr.iloc[-1] else None
        out["cmf20"] = round(float(cmf.iloc[-1]), 6) if cmf is not None and cmf.iloc[-1] == cmf.iloc[-1] else None
        out["obv"] = round(float(obv.iloc[-1]), 3) if obv is not None and obv.iloc[-1] == obv.iloc[-1] else None
        if adx is not None and len(adx):
            adx_col = next((col for col in adx.columns if str(col).startswith("ADX_")), None)
            dmp_col = next((col for col in adx.columns if str(col).startswith("DMP_")), None)
            dmn_col = next((col for col in adx.columns if str(col).startswith("DMN_")), None)
            out["adx14"] = round(float(adx[adx_col].iloc[-1]), 4) if adx_col else None
            out["di_plus"] = round(float(adx[dmp_col].iloc[-1]), 4) if dmp_col else None
            out["di_minus"] = round(float(adx[dmn_col].iloc[-1]), 4) if dmn_col else None
    except Exception as exc:
        out.update({"status": "error", "error": type(exc).__name__})
    return out


def pyvsmc_snapshot(rows: list[XAUBar]) -> dict[str, Any]:
    if len(rows) < 30:
        return {"status": "insufficient_data", "library": "pyvsmc"}
    try:
        import numpy as np
        import pyvsmc as smc
    except Exception as exc:
        return {"status": "unavailable", "library": "pyvsmc", "error": type(exc).__name__}

    open_ = np.asarray([float(r.open) for r in rows], dtype=float)
    high = np.asarray([float(r.high) for r in rows], dtype=float)
    low = np.asarray([float(r.low) for r in rows], dtype=float)
    close = np.asarray([float(r.close) for r in rows], dtype=float)
    out: dict[str, Any] = {"status": "ok", "library": "pyvsmc"}
    try:
        swings = smc.detect_swings(high, low, window_size=3, tie="first")
        structure = smc.detect_structure(
            high,
            low,
            close,
            window_size=3,
            break_mode="close",
            tie="first",
            swing_high=getattr(swings, "swing_high", None),
            swing_low=getattr(swings, "swing_low", None),
        )
        liquidity = smc.detect_liquidity(
            high,
            low,
            close,
            equal_threshold=0.001,
            sweep_lookback=30,
            window_size=3,
            tie="first",
            swing_high=getattr(swings, "swing_high", None),
            swing_low=getattr(swings, "swing_low", None),
        )
        fvg = smc.detect_fvg(high, low, close=close, compute_mitigation=True)
        obs = smc.detect_order_blocks(
            open_,
            high,
            low,
            close,
            lookback=12,
            zone_mode="body",
            compute_mitigation=True,
            tie="first",
            break_mode="close",
        )
        zones = smc.detect_zones(
            high,
            low,
            close,
            window_size=3,
            eq_threshold=0.02,
            tie="first",
            swing_high=getattr(swings, "swing_high", None),
            swing_low=getattr(swings, "swing_low", None),
        )

        bos_bull_idx, _ = _last_signal(getattr(structure, "bos_bullish", []))
        bos_bear_idx, _ = _last_signal(getattr(structure, "bos_bearish", []))
        choch_bull_idx, _ = _last_signal(getattr(structure, "choch_bullish", []))
        choch_bear_idx, _ = _last_signal(getattr(structure, "choch_bearish", []))
        sweep_high_idx, _ = _last_signal(getattr(liquidity, "sweep_high", []))
        sweep_low_idx, _ = _last_signal(getattr(liquidity, "sweep_low", []))
        bull_fvg_idx, _ = _last_signal(getattr(fvg, "bullish", []))
        bear_fvg_idx, _ = _last_signal(getattr(fvg, "bearish", []))
        bull_ob_idx, _ = _last_signal(getattr(obs, "bullish_ob", []))
        bear_ob_idx, _ = _last_signal(getattr(obs, "bearish_ob", []))

        latest_bull_structure = max([x for x in (bos_bull_idx, choch_bull_idx) if x is not None], default=-1)
        latest_bear_structure = max([x for x in (bos_bear_idx, choch_bear_idx) if x is not None], default=-1)
        direction = "bullish" if latest_bull_structure > latest_bear_structure else "bearish" if latest_bear_structure > latest_bull_structure else "neutral"

        out.update({
            "direction": direction,
            "bos": {
                "bullish_index": bos_bull_idx,
                "bearish_index": bos_bear_idx,
            },
            "choch": {
                "bullish_index": choch_bull_idx,
                "bearish_index": choch_bear_idx,
            },
            "liquidity": {
                "buy_side_sweep_index": sweep_high_idx,
                "sell_side_sweep_index": sweep_low_idx,
            },
            "fvg": {
                "bullish_index": bull_fvg_idx,
                "bearish_index": bear_fvg_idx,
            },
            "order_blocks": {
                "bullish_index": bull_ob_idx,
                "bearish_index": bear_ob_idx,
            },
            "zones": {
                "premium": bool(_native(getattr(zones, "premium", [False])[-1])),
                "discount": bool(_native(getattr(zones, "discount", [False])[-1])),
                "equilibrium": bool(_native(getattr(zones, "equilibrium", [False])[-1])),
                "in_ote": bool(_native(getattr(zones, "in_ote", [False])[-1])),
            },
        })
    except Exception as exc:
        out.update({"status": "error", "error": type(exc).__name__})
    return out


def smartmoneyconcepts_snapshot(rows: list[XAUBar]) -> dict[str, Any]:
    if len(rows) < 60:
        return {"status": "insufficient_data", "library": "smartmoneyconcepts"}
    try:
        from smartmoneyconcepts import smc
    except Exception as exc:
        return {"status": "unavailable", "library": "smartmoneyconcepts", "error": type(exc).__name__}

    df = _frame(rows).reset_index(drop=True)
    out: dict[str, Any] = {"status": "ok", "library": "smartmoneyconcepts"}
    try:
        swings = smc.swing_highs_lows(df, swing_length=10)
        structure = smc.bos_choch(df, swings, close_break=True)
        fvg = smc.fvg(df, join_consecutive=True)
        liquidity = smc.liquidity(df, swings, range_percent=0.01)
        ob = smc.ob(df, swings, close_mitigation=False)

        def last_index(frame: Any, column: str, sign: int | None = None) -> int | None:
            if frame is None or column not in frame.columns:
                return None
            series = frame[column]
            for idx in range(len(series) - 1, -1, -1):
                value = series.iloc[idx]
                if value != value:
                    continue
                if sign is None and float(value) != 0:
                    return idx
                if sign is not None and float(value) == float(sign):
                    return idx
            return None

        bos_bull = last_index(structure, "BOS", 1)
        bos_bear = last_index(structure, "BOS", -1)
        choch_bull = last_index(structure, "CHOCH", 1)
        choch_bear = last_index(structure, "CHOCH", -1)
        bull_struct = max([x for x in (bos_bull, choch_bull) if x is not None], default=-1)
        bear_struct = max([x for x in (bos_bear, choch_bear) if x is not None], default=-1)
        out.update({
            "direction": "bullish" if bull_struct > bear_struct else "bearish" if bear_struct > bull_struct else "neutral",
            "bos": {"bullish_index": bos_bull, "bearish_index": bos_bear},
            "choch": {"bullish_index": choch_bull, "bearish_index": choch_bear},
            "fvg": {
                "bullish_index": last_index(fvg, "FVG", 1),
                "bearish_index": last_index(fvg, "FVG", -1),
            },
            "liquidity": {
                "bullish_index": last_index(liquidity, "Liquidity", 1),
                "bearish_index": last_index(liquidity, "Liquidity", -1),
            },
            "order_blocks": {
                "bullish_index": last_index(ob, "OB", 1),
                "bearish_index": last_index(ob, "OB", -1),
            },
        })
    except Exception as exc:
        out.update({"status": "error", "error": type(exc).__name__})
    return out


def smc_mcp_snapshot(rows: list[XAUBar]) -> dict[str, Any]:
    if len(rows) < 30:
        return {"status": "insufficient_data", "library": "smc-mcp"}
    try:
        from smc_mcp.smc import (
            detect_structure,
            find_fair_value_gaps,
            find_liquidity_sweeps,
            find_order_blocks,
            find_swings,
        )
    except Exception as exc:
        return {"status": "unavailable", "library": "smc-mcp", "error": type(exc).__name__}

    opens = [float(r.open) for r in rows]
    highs = [float(r.high) for r in rows]
    lows = [float(r.low) for r in rows]
    closes = [float(r.close) for r in rows]
    out: dict[str, Any] = {"status": "ok", "library": "smc-mcp"}
    try:
        swings = find_swings(highs, lows, lookback=3)
        events, trend = detect_structure(closes, swings, lookback=3)
        blocks = find_order_blocks(opens, highs, lows, closes, events)
        gaps = find_fair_value_gaps(highs, lows)
        sweeps = find_liquidity_sweeps(highs, lows, closes, swings)
        out.update({
            "direction": str(trend or "neutral").lower(),
            "latest_structure": _native(events[-1]) if events else None,
            "latest_order_block": _native(blocks[-1]) if blocks else None,
            "latest_fvg": _native(gaps[-1]) if gaps else None,
            "latest_liquidity_sweep": _native(sweeps[-1]) if sweeps else None,
            "counts": {
                "structure": len(events),
                "order_blocks": len(blocks),
                "fvg": len(gaps),
                "liquidity_sweeps": len(sweeps),
            },
        })
    except Exception as exc:
        out.update({"status": "error", "error": type(exc).__name__})
    return out


def market_profile_snapshot(rows: list[XAUBar]) -> dict[str, Any]:
    if len(rows) < 30:
        return {"status": "insufficient_data", "library": "MarketProfile"}
    try:
        from market_profile import MarketProfile
        import pandas as pd
    except Exception as exc:
        return {"status": "unavailable", "library": "MarketProfile", "error": type(exc).__name__}

    try:
        df = pd.DataFrame(
            {
                "Open": [float(r.open) for r in rows],
                "High": [float(r.high) for r in rows],
                "Low": [float(r.low) for r in rows],
                "Close": [float(r.close) for r in rows],
                "Volume": [float(r.volume or 0.0) for r in rows],
            },
            index=pd.DatetimeIndex([r.timestamp for r in rows]),
        )
        if float(df["Volume"].sum()) <= 0:
            return {"status": "insufficient_volume", "library": "MarketProfile"}
        profile = MarketProfile(df)[df.index.min():df.index.max()]
        value_area = profile.value_area
        hvn = getattr(profile, "high_value_nodes", None)
        lvn = getattr(profile, "low_value_nodes", None)
        return {
            "status": "ok",
            "library": "MarketProfile",
            "poc": round(float(profile.poc_price), 6),
            "val": round(float(value_area[0]), 6),
            "vah": round(float(value_area[1]), 6),
            "hvn": [round(float(v), 6) for v in list(getattr(hvn, "index", []))[:6]],
            "lvn": [round(float(v), 6) for v in list(getattr(lvn, "index", []))[:6]],
        }
    except Exception as exc:
        return {"status": "error", "library": "MarketProfile", "error": type(exc).__name__}


def library_consensus(rows: list[XAUBar]) -> dict[str, Any]:
    primary = pyvsmc_snapshot(rows)
    oracle = smartmoneyconcepts_snapshot(rows)
    mcp_oracle = smc_mcp_snapshot(rows)
    ta = pandas_ta_snapshot(rows)
    profile = market_profile_snapshot(rows)

    directions = []
    for item in (primary, oracle, mcp_oracle):
        direction = str(item.get("direction") or "neutral")
        if item.get("status") == "ok" and direction in {"bullish", "bearish"}:
            directions.append(direction)
    bullish = sum(1 for d in directions if d == "bullish")
    bearish = sum(1 for d in directions if d == "bearish")
    consensus_direction = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
    agreement = max(bullish, bearish) / len(directions) if directions else 0.0

    return {
        "version": "library-consensus-v1",
        "direction": consensus_direction,
        "agreement": round(agreement, 4),
        "independent_direction_votes": len(directions),
        "primary": primary,
        "oracle_smc": oracle,
        "oracle_smc_mcp": mcp_oracle,
        "technical_oracle": ta,
        "profile_oracle": profile,
        "status": {
            "pyvsmc": primary.get("status"),
            "smartmoneyconcepts": oracle.get("status"),
            "smc_mcp": mcp_oracle.get("status"),
            "pandas_ta_classic": ta.get("status"),
            "marketprofile": profile.get("status"),
        },
    }


def vectorbt_validation(rows: list[XAUBar]) -> dict[str, Any]:
    """On-demand research harness; never used as a live-entry oracle."""
    if len(rows) < 220:
        return {"status": "insufficient_data", "library": "vectorbt", "minimum_bars": 220}
    try:
        import pandas as pd
        import vectorbt as vbt
    except Exception as exc:
        return {"status": "unavailable", "library": "vectorbt", "error": type(exc).__name__}

    try:
        close = pd.Series(
            [float(r.close) for r in rows],
            index=pd.DatetimeIndex([r.timestamp for r in rows]),
            name="XAUUSD",
        )
        results: list[dict[str, Any]] = []
        for fast, slow in ((9, 21), (21, 50), (50, 200)):
            fast_ma = vbt.MA.run(close, fast)
            slow_ma = vbt.MA.run(close, slow)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=10000.0, fees=0.0001)
            results.append({
                "fast": fast,
                "slow": slow,
                "total_return": round(float(pf.total_return()), 6),
                "max_drawdown": round(float(pf.max_drawdown()), 6),
                "trade_count": int(pf.trades.count()),
            })
        best = max(results, key=lambda x: x["total_return"]) if results else None
        return {
            "status": "ok",
            "library": "vectorbt",
            "research_only": True,
            "bars": len(rows),
            "strategies": results,
            "best_by_total_return": best,
        }
    except Exception as exc:
        return {"status": "error", "library": "vectorbt", "error": type(exc).__name__}
