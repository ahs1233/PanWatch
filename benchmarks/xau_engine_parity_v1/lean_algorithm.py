"""LEAN runtime side of the frozen PanWatch XAU parity benchmark."""

from AlgorithmImports import *
import hashlib
import json
from datetime import datetime


class PanWatchXAUParityAlgorithm(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2026, 1, 5)
        self.set_end_date(2026, 1, 5)
        self.set_cash(100000)

        root = "/workspace/benchmarks/xau_engine_parity_v1"
        with open(root + "/frozen_dataset.json", "r", encoding="utf-8") as handle:
            dataset = json.load(handle)
        with open(root + "/generated/lean-package.json", "r", encoding="utf-8") as handle:
            package = json.load(handle)

        if package["execution"]["live"] is not False:
            raise ValueError("LEAN parity benchmark refuses live=true")
        if package["execution"]["broker_orders_enabled"] is not False:
            raise ValueError("LEAN parity benchmark refuses broker orders")

        spec = package["strategy"]
        frames = {}
        for timeframe in spec["timeframes"]:
            frames[timeframe] = self._frame_state(
                timeframe,
                dataset["bars"][timeframe],
                spec["indicators"],
            )

        one = frames["1m"]
        five = frames["5m"]
        fifteen = frames["15m"]
        candidate = "none"
        if (
            five["direction"] == "bullish"
            and fifteen["direction"] == "bullish"
            and one["direction"] != "bearish"
        ):
            candidate = "long_setup"
        elif (
            five["direction"] == "bearish"
            and fifteen["direction"] == "bearish"
            and one["direction"] != "bullish"
        ):
            candidate = "short_setup"

        canonical = json.dumps(
            dataset,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        result = {
            "backend": "lean",
            "dataset_version": dataset["dataset_version"],
            "dataset_fingerprint": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "strategy_fingerprint": package["strategy_fingerprint"],
            "status": "ready",
            "candidate": candidate,
            "blocked": False,
            "block_reasons": [],
            "frame_states": frames,
        }
        self.debug("PANWATCH_PARITY_RESULT=" + json.dumps(result, sort_keys=True, separators=(",", ":")))
        self.quit()

    def _frame_state(self, timeframe, rows, indicators):
        fast_period = int(indicators["ema_fast"])
        slow_period = int(indicators["ema_slow"])
        rsi_period = int(indicators["rsi_period"])
        atr_period = int(indicators["atr_period"])
        lookback = int(indicators["breakout_lookback"])

        ema_fast = ExponentialMovingAverage(fast_period)
        ema_slow = ExponentialMovingAverage(slow_period)
        rsi = RelativeStrengthIndex(rsi_period, MovingAverageType.SIMPLE)
        atr = AverageTrueRange(atr_period, MovingAverageType.SIMPLE)
        symbol = Symbol.create("XAUUSD", SecurityType.CFD, Market.OANDA)

        for row in rows:
            time = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).replace(tzinfo=None)
            close = float(row["close"])
            point = IndicatorDataPoint(time, close)
            ema_fast.update(point)
            ema_slow.update(point)
            rsi.update(point)
            atr.update(
                TradeBar(
                    time,
                    symbol,
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    close,
                    float(row["volume"]),
                )
            )

        latest = rows[-1]
        latest_close = float(latest["close"])
        prior = rows[-(lookback + 1):-1]
        prior_high = max(float(row["high"]) for row in prior)
        prior_low = min(float(row["low"]) for row in prior)
        breakout = (
            "up"
            if latest_close > prior_high
            else "down"
            if latest_close < prior_low
            else "none"
        )

        fast = float(ema_fast.current.value)
        slow = float(ema_slow.current.value)
        rsi_value = float(rsi.current.value)
        score = 0
        if latest_close > fast > slow:
            score += 1
        elif latest_close < fast < slow:
            score -= 1
        if rsi_value >= 55:
            score += 1
        elif rsi_value <= 45:
            score -= 1
        if breakout == "up":
            score += 1
        elif breakout == "down":
            score -= 1
        direction = "bullish" if score >= 2 else "bearish" if score <= -2 else "neutral"

        return {
            "direction": direction,
            "breakout": breakout,
            "ema_fast": round(fast, 10),
            "ema_slow": round(slow, 10),
            "rsi14": round(rsi_value, 10),
            "atr14": round(float(atr.current.value), 10),
            "close": round(latest_close, 10),
        }
