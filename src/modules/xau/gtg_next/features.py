from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


class FeatureEngine:
    """GTG-specific composition of mature indicator primitives.

    Custom code here only defines GTG semantics (slope/curvature/spacing);
    EMA and RSI calculations are delegated to pandas-ta-classic.
    """

    def __init__(
        self,
        ema_lengths: tuple[int, ...] = (14, 22, 50, 200),
        rsi_length: int = 14,
    ) -> None:
        self.ema_lengths = ema_lengths
        self.rsi_length = rsi_length

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing OHLC columns: {sorted(missing)}")
        if not frame.index.is_monotonic_increasing:
            raise ValueError("Input index must be monotonic increasing")

        out = pd.DataFrame(index=frame.index)
        close = frame["close"].astype(float)

        for length in self.ema_lengths:
            ema = ta.ema(close, length=length)
            if ema is None:
                raise RuntimeError(f"pandas-ta-classic failed EMA({length})")
            name = f"ema_{length}"
            out[name] = ema
            out[f"{name}_slope"] = out[name].diff()
            out[f"{name}_curvature"] = out[f"{name}_slope"].diff()
            out[f"price_to_{name}_pct"] = (close / out[name]) - 1.0

        rsi = ta.rsi(close, length=self.rsi_length)
        if rsi is None:
            raise RuntimeError(f"pandas-ta-classic failed RSI({self.rsi_length})")
        out["rsi"] = rsi
        out["rsi_slope"] = out["rsi"].diff()
        out["rsi_curvature"] = out["rsi_slope"].diff()

        if {"ema_14", "ema_50"}.issubset(out.columns):
            out["ema_14_50_spacing_pct"] = (
                (out["ema_14"] - out["ema_50"]) / close.replace(0, np.nan)
            )
        if {"ema_50", "ema_200"}.issubset(out.columns):
            out["ema_50_200_spacing_pct"] = (
                (out["ema_50"] - out["ema_200"]) / close.replace(0, np.nan)
            )

        out["range_pct"] = (
            (frame["high"].astype(float) - frame["low"].astype(float))
            / close.replace(0, np.nan)
        )
        out["return_1"] = close.pct_change()
        return out.replace([np.inf, -np.inf], np.nan)
