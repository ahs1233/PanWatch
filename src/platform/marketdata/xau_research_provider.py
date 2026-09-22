"""Research-only gold proxy using Yahoo Finance GC=F.

This provider is deliberately marked non-execution-eligible. It is useful for
research, chart context, and TradingAgents grounding, but it is not spot
XAUUSD and must not be used for broker entry/SL/TP prices.
"""

from __future__ import annotations

from datetime import timezone

from .xau_models import XAUBar, XAUTimeframe

_PERIOD_BY_TIMEFRAME = {
    XAUTimeframe.M1: "5d",
    XAUTimeframe.M5: "1mo",
    XAUTimeframe.M15: "1mo",
    XAUTimeframe.H1: "2y",
    XAUTimeframe.D1: "max",
    XAUTimeframe.W1: "max",
    XAUTimeframe.MN1: "max",
}

_INTERVAL_BY_TIMEFRAME = {
    XAUTimeframe.M1: "1m",
    XAUTimeframe.M5: "5m",
    XAUTimeframe.M15: "15m",
    XAUTimeframe.H1: "1h",
    XAUTimeframe.D1: "1d",
    XAUTimeframe.W1: "1wk",
    XAUTimeframe.MN1: "1mo",
}


class YahooGoldResearchProvider:
    symbol = "GC=F"
    source = "yfinance:GC=F"
    execution_eligible = False

    def bars(self, timeframe: XAUTimeframe) -> list[XAUBar]:
        try:
            import os
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for GC=F research data") from exc

        cache_dir = os.environ.get("YFINANCE_CACHE_DIR", "/tmp/panwatch-yfinance")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            yf.set_tz_cache_location(cache_dir)
        except Exception:
            pass

        ticker = yf.Ticker(self.symbol)
        if timeframe in {XAUTimeframe.D1, XAUTimeframe.W1, XAUTimeframe.MN1}:
            # Explicit start is more reliable than period="max" for some Yahoo
            # futures responses and gives enough history for daily/weekly EMA1000.
            frame = ticker.history(
                start="2000-01-01",
                interval=_INTERVAL_BY_TIMEFRAME[timeframe],
                auto_adjust=False,
                actions=False,
            )
        else:
            frame = ticker.history(
                period=_PERIOD_BY_TIMEFRAME[timeframe],
                interval=_INTERVAL_BY_TIMEFRAME[timeframe],
                auto_adjust=False,
                actions=False,
            )
        if frame is None or frame.empty:
            return []

        out: list[XAUBar] = []
        for index, row in frame.iterrows():
            timestamp = index.to_pydatetime()
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)

            try:
                open_ = float(row["Open"])
                high = float(row["High"])
                low = float(row["Low"])
                close = float(row["Close"])
                volume_raw = row.get("Volume")
                volume = float(volume_raw) if volume_raw is not None else None
                values = (open_, high, low, close)
                if any(value != value or value <= 0 for value in values):
                    continue
                if high < max(open_, close, low) or low > min(open_, close, high):
                    continue
                if volume is not None and volume != volume:
                    volume = None
                out.append(
                    XAUBar(
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=open_,
                        high=high,
                        low=low,
                        close=close,
                        volume=volume,
                        source=self.source,
                        symbol="XAUUSD",
                        execution_eligible=False,
                    )
                )
            except (TypeError, ValueError, OverflowError):
                # Long-run futures history occasionally contains malformed or
                # settlement-only rows. One bad Yahoo row must not discard the
                # entire EMA history.
                continue
        return out
