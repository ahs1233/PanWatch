from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import edge_analysis as ea

UTC = timezone.utc


def frame(start: datetime, n: int, *, direction: str = "LONG", mean: float = 4.0, session: str = "asia") -> pd.DataFrame:
    rng = np.random.default_rng(7 + n + int(start.timestamp()) % 97)
    values = rng.normal(mean, 5.0, n)
    return pd.DataFrame({
        "observed_at": [start + timedelta(hours=4*i) for i in range(n)],
        "direction": [direction] * n,
        "session": [session] * n,
        "session_transition": ["none"] * n,
        "volatility_quartile": ["q4_high"] * n,
        "regime": ["trend_bull"] * n,
        "technical_alignment": ["bullish"] * n,
        "htf_composite_direction": ["bullish"] * n,
        "today_direction": ["bullish"] * n,
        "cash_flow_direction": ["inflow"] * n,
        "smart_money_bias": ["bullish"] * n,
        "volume_profile_location": ["above_value"] * n,
        "dealing_zone": ["discount"] * n,
        "break_of_structure": ["bullish"] * n,
        "liquidity_sweep": ["sell_side_sweep"] * n,
        "displacement": ["bullish"] * n,
        "macro_bias_label": ["bullish"] * n,
        "htf_monthly_direction": ["bullish"] * n,
        "htf_weekly_direction": ["bullish"] * n,
        "htf_daily_direction": ["bullish"] * n,
        "htf_h4_direction": ["bullish"] * n,
        "htf_h1_direction": ["bullish"] * n,
        "gen1_decision": ["LONG"] * n,
        "fusion_state": ["aligned"] * n,
        "net_bps": values,
        "gross_bps": values + 0.5,
        "month": [(start + timedelta(hours=4*i)).strftime("%Y-%m") for i in range(n)],
        "cognitive_confidence": np.linspace(.55, .85, n),
        "gen1_confidence": np.linspace(.55, .85, n),
        "atr_pct_5m": np.linspace(.08, .22, n),
        "atr_pct_15m": np.linspace(.10, .30, n),
        "rsi_5m": np.linspace(48, 67, n),
        "rsi_15m": np.linspace(47, 65, n),
        "htf_composite_score": np.linspace(.1, .7, n),
        "today_score": np.linspace(.1, .7, n),
        "cash_flow_score": np.linspace(.1, .5, n),
        "smart_money_score": np.linspace(.1, .5, n),
        "macro_proxy_score": np.linspace(.05, .4, n),
        "htf_alignment_ratio": np.linspace(.6, 1.0, n),
        "htf_conflict_count": np.zeros(n),
        "dxy_5d": np.linspace(-.02, -.001, n),
        "us10y_5d_delta": np.linspace(-.4, -.01, n),
        "vix_5d": np.linspace(.01, .15, n),
        "spx_5d": np.linspace(-.03, -.001, n),
        "oil_5d": np.linspace(.001, .05, n),
        "state_trend_strength": np.linspace(.2, .9, n),
        "state_volatility": np.linspace(.2, .9, n),
        "state_directional_edge": np.linspace(.1, .7, n),
        "pm10_favorable_first": [True] * n,
        "pm10_adverse_first": [False] * n,
        "pm20_favorable_first": [True] * n,
        "pm20_adverse_first": [False] * n,
        "pm30_favorable_first": [True] * n,
        "pm30_adverse_first": [False] * n,
    })


class EdgeAnalysisTests(unittest.TestCase):
    def test_policy_lock_ignores_final_holdout(self):
        discovery = frame(datetime(2024, 9, 23, tzinfo=UTC), 120)
        validation = frame(datetime(2025, 5, 23, tzinfo=UTC), 120)
        table = ea._evaluate_rules(discovery, validation)
        policy_a = ea._lock_policy(table)
        final_bad = frame(datetime(2026, 1, 23, tzinfo=UTC), 120, mean=-30.0)
        _ = ea._apply_policy(final_bad, policy_a)
        policy_b = ea._lock_policy(table.copy())
        self.assertEqual(policy_a, policy_b)

    def test_unstable_rule_is_not_selected(self):
        discovery = frame(datetime(2024, 9, 23, tzinfo=UTC), 120, mean=6.0)
        validation = frame(datetime(2025, 5, 23, tzinfo=UTC), 120, mean=-6.0)
        table = ea._evaluate_rules(discovery, validation)
        policy = ea._lock_policy(table)
        self.assertEqual(policy["selected_rules"], [])
        applied = ea._apply_policy(validation, policy)
        self.assertTrue(applied["gen11_decision"].eq("WAIT").all())

    def test_independent_loop_matches_vectorized_policy(self):
        data = frame(datetime(2026, 1, 23, tzinfo=UTC), 80)
        rule = {"id": "LONG__session:eq:asia", "direction": "LONG", "clauses": [("session", "eq", "asia")]}
        policy = {"selected_rules": [rule]}
        parity = ea._independent_validation(data, policy)
        self.assertTrue(parity["parity"])
        self.assertEqual(parity["vectorized_count"], 80)

    def test_metrics_include_required_risk_statistics(self):
        data = frame(datetime(2026, 1, 23, tzinfo=UTC), 40)
        metrics = ea._metric_block(data)
        for key in ("mean_net_bps", "median_net_bps", "positive_rate", "wilson_95_ci", "bootstrap_mean_95_ci_bps", "max_drawdown_bps", "cumulative_net_bps"):
            self.assertIn(key, metrics)


if __name__ == "__main__":
    unittest.main()
