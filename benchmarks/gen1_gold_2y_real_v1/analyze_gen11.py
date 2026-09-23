#!/usr/bin/env python3
"""GEN1.1 conditional-edge discovery with an untouched chronological holdout."""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

try:
    import ruptures as rpt
except Exception:
    rpt = None

SEED = 20260924
START = pd.Timestamp("2024-09-23", tz="UTC")
DISCOVERY_END = pd.Timestamp("2025-09-23", tz="UTC")
VALIDATION_END = pd.Timestamp("2026-03-23", tz="UTC")
END = pd.Timestamp("2026-09-23", tz="UTC")


@dataclass(frozen=True)
class Gate:
    gate_id: str
    family: str
    description: str
    func: Callable[[pd.DataFrame], pd.Series]


def decorrelate(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    chosen: list[int] = []
    last = None
    gap = pd.Timedelta(minutes=minutes)
    for idx, timestamp in zip(df.index, df.observed_at):
        if last is None or timestamp - last >= gap:
            chosen.append(idx)
            last = timestamp
    return df.loc[chosen].copy().reset_index(drop=True)


def bootstrap_ci(values, iterations: int = 4000) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None, None
    rng = np.random.default_rng(SEED)
    means = np.empty(iterations)
    for idx in range(iterations):
        means[idx] = rng.choice(values, len(values), replace=True).mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def metrics(df: pd.DataFrame) -> dict[str, Any]:
    values = pd.to_numeric(df.net_spread_directional_bps, errors="coerce").dropna().to_numpy(float)
    if not len(values):
        return {"n": 0}
    wins = int((values > 0).sum())
    boot_lo, boot_hi = bootstrap_ci(values)
    wilson_lo, wilson_hi = proportion_confint(wins, len(values), method="wilson")
    equity = np.cumsum(values)
    peak = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    monthly = []
    tmp = df.assign(_value=pd.to_numeric(df.net_spread_directional_bps, errors="coerce")).dropna(subset=["_value"])
    for month, group in tmp.groupby(tmp.observed_at.dt.strftime("%Y-%m")):
        monthly.append({"month": month, "n": len(group), "mean_bps": round(float(group._value.mean()), 4)})
    return {
        "n": int(len(values)),
        "mean_net_bps": round(float(values.mean()), 4),
        "median_net_bps": round(float(np.median(values)), 4),
        "positive_rate": round(float((values > 0).mean()), 4),
        "wilson_95_ci": [round(float(wilson_lo), 4), round(float(wilson_hi), 4)],
        "bootstrap_mean_95_ci_bps": [round(float(boot_lo), 4), round(float(boot_hi), 4)],
        "cumulative_net_bps": round(float(values.sum()), 4),
        "max_drawdown_bps": round(float(np.min(equity - peak)), 4),
        "profitable_months": sum(item["mean_bps"] > 0 for item in monthly),
        "months": len(monthly),
        "monthly": monthly,
    }


def split(df: pd.DataFrame):
    return (
        df[(df.observed_at >= START) & (df.observed_at < DISCOVERY_END)].copy(),
        df[(df.observed_at >= DISCOVERY_END) & (df.observed_at < VALIDATION_END)].copy(),
        df[(df.observed_at >= VALIDATION_END) & (df.observed_at < END)].copy(),
    )


def prepare(df: pd.DataFrame, thresholds: dict[str, list[float]] | None = None):
    out = df.copy().sort_values("observed_at").reset_index(drop=True)
    out["candidate_side"] = np.where(out.candidate.eq("long_setup"), 1.0, -1.0)
    out["hour"] = out.observed_at.dt.hour
    out["weekday"] = out.observed_at.dt.dayofweek
    out["month"] = out.observed_at.dt.month
    aligned = [
        "htf_composite_score", "htf_today_score", "macro_proxy_score", "flow_score", "smart_score", "evidence_score",
        "monthly_score", "weekly_score", "daily_score", "h4_score", "h1_score",
        "macro_dxy_5d_gold_score", "macro_vix_5d_gold_score", "macro_spx_5d_gold_score",
        "macro_oil_5d_gold_score", "macro_us10y_5d_delta_gold_score",
    ]
    for col in aligned:
        if col in out:
            out["aligned_" + col] = pd.to_numeric(out[col], errors="coerce") * out.candidate_side
    for timeframe in ("monthly", "weekly", "daily", "h4", "h1"):
        for period in (9, 21, 50, 200, 1000):
            col = f"{timeframe}_ema{period}_distance_pct"
            if col in out:
                out["aligned_" + col] = pd.to_numeric(out[col], errors="coerce") * out.candidate_side
    out["trend_strength"] = pd.to_numeric(out.get("htf_composite_score"), errors="coerce").abs()
    atr_cols = [col for col in ("5m_atr_pct", "15m_atr_pct") if col in out]
    out["volatility_pct"] = out[atr_cols].apply(pd.to_numeric, errors="coerce").max(axis=1) if atr_cols else np.nan

    fitted = thresholds is None
    thresholds = dict(thresholds or {})
    discovery = out[(out.observed_at >= START) & (out.observed_at < DISCOVERY_END)]
    quartile_inputs = [
        "volatility_pct", "trend_strength", "cognitive_confidence", "gen1_confidence",
        "evidence_conflict_score", "evidence_directional_confidence",
    ]
    for col in quartile_inputs:
        if col not in out:
            continue
        if fitted:
            series = pd.to_numeric(discovery[col], errors="coerce").dropna()
            if len(series) < 50:
                continue
            qs = [float(value) for value in series.quantile([0.25, 0.5, 0.75])]
            if len(set(qs)) < 3:
                continue
            thresholds[col] = qs
        if col in thresholds:
            out[col + "_quartile"] = pd.cut(
                pd.to_numeric(out[col], errors="coerce"),
                [-np.inf, *thresholds[col], np.inf],
                labels=["q1", "q2", "q3", "q4"],
            ).astype(str)
    return out, thresholds


def build_gates(discovery: pd.DataFrame) -> list[Gate]:
    gates: list[Gate] = []

    def categorical(col: str, values: list[str]):
        if col not in discovery:
            return
        for value in values:
            gates.append(Gate(
                f"{col}={value}", col, f"{col} == {value}",
                lambda frame, col=col, value=value: frame[col].astype(str).eq(value),
            ))

    categorical("session", ["asia", "london", "new_york", "off_hours"])
    categorical("regime", ["trend_bull", "trend_bear", "compression_range", "breakout_expansion"])
    categorical("alignment", ["bullish", "bearish", "mixed"])
    categorical("session_transition", ["True", "False"])
    categorical("1m_direction", ["bullish", "bearish", "neutral"])
    for col in ("1m_breakout", "5m_breakout", "15m_breakout"):
        categorical(col, ["up", "down", "none"])
    for col in ("htf_composite_direction", "htf_today_direction", "monthly_direction", "weekly_direction", "daily_direction", "h4_direction", "h1_direction"):
        categorical(col, ["bullish", "bearish", "neutral"])
    categorical("flow_direction", ["inflow", "outflow", "balanced"])
    categorical("smart_bias", ["bullish", "bearish", "neutral"])
    categorical("smart_break_of_structure", ["bullish", "bearish", "none"])
    categorical("smart_liquidity_sweep", ["buy_side_sweep", "sell_side_sweep", "none"])
    categorical("smart_displacement", ["bullish", "bearish", "none"])
    categorical("smart_dealing_zone", ["premium", "discount", "equilibrium"])
    categorical("volume_location", ["above_value", "inside_value", "below_value"])
    categorical("evidence_direction", ["bullish", "bearish", "neutral"])
    for col in [name for name in discovery.columns if name.endswith("_quartile")]:
        categorical(col, ["q1", "q2", "q3", "q4"])

    numeric = [
        "aligned_htf_composite_score", "aligned_htf_today_score", "aligned_macro_proxy_score",
        "aligned_flow_score", "aligned_smart_score", "aligned_evidence_score", "aligned_monthly_score",
        "aligned_weekly_score", "aligned_daily_score", "aligned_h4_score", "aligned_h1_score",
        "5m_rsi14", "15m_rsi14", "5m_atr_pct", "15m_atr_pct", "evidence_coverage",
        "evidence_agreement_ratio", "evidence_conflict_score", "evidence_directional_confidence",
        "cognitive_confidence", "gen1_confidence", "flow_cmf20", "flow_signed_tick_volume_imbalance",
        "liquidity_nearest_distance", "trend_strength",
    ]
    numeric += [name for name in discovery.columns if name.startswith("aligned_") and ("_ema" in name or name.startswith("aligned_macro_"))]
    numeric += [name for name in discovery.columns if name.endswith("_swing_high_distance_atr") or name.endswith("_swing_low_distance_atr")]
    for col in dict.fromkeys(numeric):
        if col not in discovery:
            continue
        series = pd.to_numeric(discovery[col], errors="coerce").dropna()
        if len(series) < 100 or series.nunique() < 10:
            continue
        q25, q50, q75 = [float(value) for value in series.quantile([0.25, 0.5, 0.75])]
        for label, op, threshold in (("ge_q75", "ge", q75), ("ge_q50", "ge", q50), ("le_q25", "le", q25)):
            gates.append(Gate(
                f"{col}:{label}:{threshold:.8g}", col, f"{col} {op} {threshold:.6g}",
                lambda frame, col=col, threshold=threshold, op=op: (
                    pd.to_numeric(frame[col], errors="coerce").ge(threshold)
                    if op == "ge" else pd.to_numeric(frame[col], errors="coerce").le(threshold)
                ),
            ))
    return gates


def allowed_pairs(gates: list[Gate]):
    interaction = {
        "session": {"regime", "alignment", "htf_composite_direction", "volatility_pct_quartile", "aligned_htf_composite_score", "aligned_macro_proxy_score", "aligned_flow_score", "aligned_smart_score", "evidence_direction"},
        "regime": {"alignment", "htf_composite_direction", "volatility_pct_quartile", "aligned_htf_composite_score", "aligned_macro_proxy_score", "aligned_flow_score", "aligned_smart_score", "evidence_direction"},
        "alignment": {"volatility_pct_quartile", "aligned_macro_proxy_score", "aligned_flow_score", "aligned_smart_score", "evidence_direction"},
        "weekly_direction": {"daily_direction", "session", "volatility_pct_quartile", "evidence_direction"},
        "daily_direction": {"h4_direction", "session", "volatility_pct_quartile", "evidence_direction"},
    }
    pairs = []
    for idx, left in enumerate(gates):
        for right in gates[idx + 1:]:
            if left.family == right.family:
                continue
            if right.family in interaction.get(left.family, set()) or left.family in interaction.get(right.family, set()):
                pairs.append((left, right))
    return pairs


def test_rule(base: pd.DataFrame, func):
    mask = func(base).fillna(False)
    selected = pd.to_numeric(base.loc[mask, "net_spread_directional_bps"], errors="coerce").dropna().to_numpy(float)
    rejected = pd.to_numeric(base.loc[~mask, "net_spread_directional_bps"], errors="coerce").dropna().to_numpy(float)
    if len(selected) < 60 or len(rejected) < 60:
        return len(selected), np.nan, np.nan, np.nan
    pvalue = float(mannwhitneyu(selected, rejected, alternative="greater").pvalue)
    return len(selected), float(selected.mean()), float((selected > 0).mean()), pvalue


def discover(discovery: pd.DataFrame, validation: pd.DataFrame):
    gates = build_gates(discovery)
    rules: list[dict[str, Any]] = []
    funcs: dict[str, Callable] = {}
    for side, candidate in (("LONG", "long_setup"), ("SHORT", "short_setup")):
        base = discovery[discovery.candidate.eq(candidate)]
        tests = [(gate.gate_id, gate.description, [gate.family], gate.func) for gate in gates]
        for left, right in allowed_pairs(gates):
            tests.append((
                left.gate_id + "&" + right.gate_id,
                left.description + " AND " + right.description,
                [left.family, right.family],
                lambda frame, left=left, right=right: left.func(frame) & right.func(frame),
            ))
        for gate_id, description, families, func in tests:
            n, mean, positive, pvalue = test_rule(base, func)
            if not np.isfinite(pvalue):
                continue
            rule_id = side + "|" + gate_id
            funcs[rule_id] = func
            rules.append({"rule_id": rule_id, "side": side, "description": description, "families": families, "n": n, "mean": mean, "positive_rate": positive, "p": pvalue})
    if rules:
        reject, qvalues, _, _ = multipletests([row["p"] for row in rules], alpha=0.10, method="fdr_bh")
        for row, accepted, qvalue in zip(rules, reject, qvalues):
            row["q"] = float(qvalue)
            row["fdr_reject"] = bool(accepted)
    for row in rules:
        if not row.get("fdr_reject") or row["mean"] <= 0:
            continue
        candidate = "long_setup" if row["side"] == "LONG" else "short_setup"
        base = validation[validation.candidate.eq(candidate)]
        subset = base[funcs[row["rule_id"]](base).fillna(False)]
        row["validation"] = metrics(subset)
        row["validation_pass"] = row["validation"].get("n", 0) >= 30 and (row["validation"].get("mean_net_bps") or -1e9) > 0
    return rules, funcs


def lock_policy(rules: list[dict[str, Any]]) -> dict[str, str | None]:
    policy: dict[str, str | None] = {"LONG": None, "SHORT": None}
    for side in policy:
        eligible = [row for row in rules if row["side"] == side and row.get("validation_pass")]
        eligible.sort(key=lambda row: (
            ((row["validation"].get("bootstrap_mean_95_ci_bps") or [-1e9])[0] or -1e9),
            row["validation"].get("mean_net_bps") or -1e9,
            row["validation"].get("n", 0),
            -row.get("q", 1.0),
        ), reverse=True)
        if eligible:
            policy[side] = eligible[0]["rule_id"]
    return policy


def apply_policy(df: pd.DataFrame, policy: dict[str, str | None], funcs: dict[str, Callable]):
    out = df.copy()
    out["gen11_decision"] = "WAIT"
    for side, candidate in (("LONG", "long_setup"), ("SHORT", "short_setup")):
        rule_id = policy.get(side)
        if rule_id:
            mask = out.candidate.eq(candidate) & funcs[rule_id](out).fillna(False)
            out.loc[mask, "gen11_decision"] = side
    return out


def directional(df: pd.DataFrame, column: str) -> dict[str, Any]:
    return metrics(df[df[column].isin(["LONG", "SHORT"])])


def diagnostics(discovery: pd.DataFrame, validation: pd.DataFrame):
    numeric = [name for name in discovery.columns if name.startswith("aligned_")]
    numeric += [name for name in ("volatility_pct", "trend_strength", "1m_rsi14", "5m_rsi14", "15m_rsi14", "1m_atr_pct", "5m_atr_pct", "15m_atr_pct", "1m_fast_distance_atr", "5m_fast_distance_atr", "15m_fast_distance_atr", "evidence_coverage", "evidence_agreement_ratio", "evidence_conflict_score", "evidence_directional_confidence", "flow_cmf20") if name in discovery]
    numeric = [name for name in dict.fromkeys(numeric) if pd.to_numeric(discovery[name], errors="coerce").notna().sum() >= 100]
    categorical = [name for name in ("session", "regime", "alignment", "flow_direction", "smart_bias", "volume_location", "evidence_direction") if name in discovery]
    result: dict[str, Any] = {}
    if numeric or categorical:
        transformer = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])
        model = Pipeline([("prep", transformer), ("clf", LogisticRegression(C=0.35, solver="liblinear", max_iter=2000, random_state=SEED))])
        xd = discovery[numeric + categorical]
        yd = (pd.to_numeric(discovery.net_spread_directional_bps, errors="coerce") > 0).astype(int)
        xv = validation[numeric + categorical]
        yv = (pd.to_numeric(validation.net_spread_directional_bps, errors="coerce") > 0).astype(int)
        model.fit(xd, yd)
        probabilities = model.predict_proba(xv)[:, 1]
        permutation = permutation_importance(model, xv, yv, scoring="roc_auc", n_repeats=10, random_state=SEED)
        result["logistic"] = {
            "validation_auc": round(float(roc_auc_score(yv, probabilities)), 4),
            "validation_brier": round(float(brier_score_loss(yv, probabilities)), 4),
            "permutation_importance": sorted([
                {"feature": name, "importance_mean": round(float(mean), 6), "importance_std": round(float(std), 6)}
                for name, mean, std in zip(numeric + categorical, permutation.importances_mean, permutation.importances_std)
            ], key=lambda item: item["importance_mean"], reverse=True)[:20],
        }
    ols_columns = [name for name in numeric if name.startswith("aligned_")]
    if ols_columns:
        tmp = discovery[["net_spread_directional_bps", *ols_columns]].copy()
        tmp["net_spread_directional_bps"] = pd.to_numeric(tmp.net_spread_directional_bps, errors="coerce")
        tmp = tmp.dropna(subset=["net_spread_directional_bps"])
        design = tmp[ols_columns].apply(pd.to_numeric, errors="coerce")
        design = design.fillna(design.median())
        design = (design - design.mean()) / design.std().replace(0, 1)
        fit = OLS(tmp.net_spread_directional_bps, add_constant(design)).fit(cov_type="HC3")
        rows = [{"feature": name, "coef": float(fit.params[name]), "t": float(fit.tvalues[name]), "p": float(fit.pvalues[name])} for name in ols_columns]
        if rows:
            _, qvalues, _, _ = multipletests([row["p"] for row in rows], alpha=0.10, method="fdr_bh")
            for row, qvalue in zip(rows, qvalues):
                row["q"] = float(qvalue)
        rows.sort(key=lambda row: abs(row["t"]), reverse=True)
        result["robust_ols"] = {"n": int(fit.nobs), "r2": round(float(fit.rsquared), 4), "top": rows[:20]}
    return result


def regime_discovery(discovery: pd.DataFrame):
    if rpt is None:
        return {"available": False, "reason": "ruptures_not_installed"}
    columns = [name for name in ("aligned_htf_composite_score", "volatility_pct", "aligned_macro_proxy_score", "aligned_flow_score", "aligned_smart_score") if name in discovery]
    if len(columns) < 2 or len(discovery) < 150:
        return {"available": False, "reason": "insufficient_features"}
    matrix = discovery[columns].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median()).to_numpy(float)
    scale = matrix.std(axis=0)
    scale[scale == 0] = 1.0
    matrix = (matrix - matrix.mean(axis=0)) / scale
    penalty = max(6.0, 3.0 * math.log(len(matrix)))
    breaks = rpt.Pelt(model="rbf", min_size=30, jump=5).fit(matrix).predict(pen=penalty)
    points = [int(index) for index in breaks[:-1] if 0 < index <= len(discovery)]
    boundaries = [0, *points, len(discovery)]
    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        segment = discovery.iloc[left:right]
        if segment.empty:
            continue
        segments.append({
            "start": segment.observed_at.iloc[0].isoformat(),
            "end": segment.observed_at.iloc[-1].isoformat(),
            "n": len(segment),
            "candidate_metrics": metrics(segment),
            "current_regime_counts": segment.regime.value_counts().to_dict(),
        })
    return {"available": True, "method": "ruptures.Pelt(rbf)", "penalty": round(penalty, 4), "features": columns, "change_points": [discovery.iloc[index - 1].observed_at.isoformat() for index in points], "segments": segments, "used_for_policy": False}


def subgroup_report(df: pd.DataFrame):
    output = {}
    for col in ("candidate", "session", "regime", "gen1_decision", "fusion_state", "alignment", "flow_direction", "smart_bias", "volume_location", "evidence_direction", "session_transition", "volatility_pct_quartile", "weekday", "hour", "month"):
        if col not in df:
            continue
        rows = []
        for value, group in df.groupby(col, dropna=False):
            summary = metrics(group)
            if summary.get("n", 0) >= 20:
                rows.append({"value": str(value), **{key: val for key, val in summary.items() if key != "monthly"}})
        rows.sort(key=lambda row: row.get("mean_net_bps") or -1e9, reverse=True)
        output[col] = rows
    return output


def range_report(df: pd.DataFrame):
    output = {}
    for distance in (10, 20, 30):
        col = f"range_pm{distance}_favorable_first"
        if col not in df:
            continue
        values = df[col].astype(str).str.lower()
        resolved = values.isin(["true", "false"])
        count = int(resolved.sum())
        wins = int((values[resolved] == "true").sum())
        output[f"pm{distance}"] = {"resolved_n": count, "favorable_first_rate": round(wins / count, 4) if count else None}
    return output


def temporal_folds(df: pd.DataFrame, decision_column: str, folds: int = 3):
    if df.empty:
        return []
    boundaries = pd.date_range(df.observed_at.min(), df.observed_at.max() + pd.Timedelta(seconds=1), periods=folds + 1)
    rows = []
    for index in range(folds):
        lo, hi = boundaries[index], boundaries[index + 1]
        summary = directional(df[(df.observed_at >= lo) & (df.observed_at < hi)], decision_column)
        rows.append({"fold": index + 1, "start": lo.isoformat(), "end": hi.isoformat(), **{key: val for key, val in summary.items() if key != "monthly"}})
    return rows


def evaluate_horizon(raw: pd.DataFrame, minutes: int, thresholds, policy, funcs):
    independent = decorrelate(raw.sort_values("observed_at").reset_index(drop=True), minutes)
    prepared, _ = prepare(independent, thresholds=thresholds)
    decided = apply_policy(prepared, policy, funcs)
    _, _, final = split(decided)
    return {
        "independent_n": len(decided),
        "full_period": {"gen1": directional(decided, "gen1_decision"), "gen11": directional(decided, "gen11_decision")},
        "final_holdout": {"gen1": directional(final, "gen1_decision"), "gen11": directional(final, "gen11_decision")},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_240m")
    parser.add_argument("--csv-60m")
    parser.add_argument("--out", required=True)
    parser.add_argument("--policy-out")
    args = parser.parse_args()

    raw240 = pd.read_csv(args.csv_240m, parse_dates=["observed_at", "outcome_at"]).sort_values("observed_at").reset_index(drop=True)
    independent240 = decorrelate(raw240, 240)
    prepared240, thresholds = prepare(independent240)
    discovery, validation, _ = split(prepared240)
    rules, funcs = discover(discovery, validation)
    policy = lock_policy(rules)  # Locked before final holdout metrics are evaluated.
    decided240 = apply_policy(prepared240, policy, funcs)
    discovery_decided, validation_decided, final240 = split(decided240)

    report = {
        "version": "gen1.1-edge-decomposition-v1",
        "research_only": True,
        "execution_allowed": False,
        "edge_proven": False,
        "splits": {
            "discovery": [START.isoformat(), DISCOVERY_END.isoformat()],
            "validation": [DISCOVERY_END.isoformat(), VALIDATION_END.isoformat()],
            "final_holdout": [VALIDATION_END.isoformat(), END.isoformat()],
        },
        "raw_episode_n": len(raw240),
        "independent_240m_n": len(prepared240),
        "discovery_n": len(discovery),
        "validation_n": len(validation),
        "final_holdout_n": len(final240),
        "multiple_testing": {"tested_hypotheses": len(rules), "method": "Benjamini-Hochberg FDR 10%", "final_holdout_used_for_selection": False},
        "discovery_quantiles": thresholds,
        "locked_policy": policy,
        "selected_rules": [row for row in rules if row["rule_id"] in {value for value in policy.values() if value}],
        "top_hypotheses": sorted(rules, key=lambda row: (row.get("q", 1.0), -(row.get("mean") or -1e9)))[:60],
        "models": diagnostics(discovery, validation),
        "regime_discovery": regime_discovery(discovery),
        "subgroups": {"discovery": subgroup_report(discovery_decided), "validation": subgroup_report(validation_decided), "final_holdout": subgroup_report(final240)},
        "range_outcomes": {"discovery": range_report(discovery_decided), "validation": range_report(validation_decided), "final_holdout": range_report(final240)},
        "horizon_240m": {
            "full_period": {"gen1": directional(decided240, "gen1_decision"), "gen11": directional(decided240, "gen11_decision")},
            "final_holdout": {"gen1": directional(final240, "gen1_decision"), "gen11": directional(final240, "gen11_decision")},
            "final_holdout_folds": {"gen1": temporal_folds(final240, "gen1_decision"), "gen11": temporal_folds(final240, "gen11_decision")},
        },
    }
    if args.csv_60m:
        raw60 = pd.read_csv(args.csv_60m, parse_dates=["observed_at", "outcome_at"])
        report["horizon_60m"] = evaluate_horizon(raw60, 60, thresholds, policy, funcs)

    final_metrics = report["horizon_240m"]["final_holdout"]["gen11"]
    confidence_interval = final_metrics.get("bootstrap_mean_95_ci_bps") or [None, None]
    folds = report["horizon_240m"]["final_holdout_folds"]["gen11"]
    profitable_folds = sum(1 for fold in folds if (fold.get("mean_net_bps") or 0) > 0)
    report["edge_proven"] = bool(
        final_metrics.get("n", 0) >= 50
        and (final_metrics.get("mean_net_bps") or 0) > 0
        and confidence_interval[0] is not None
        and confidence_interval[0] > 0
        and profitable_folds >= 2
        and final_metrics.get("profitable_months", 0) >= max(2, math.ceil(0.5 * max(1, final_metrics.get("months", 0))))
    )
    report["final_holdout_fold_gate"] = {"profitable_folds": profitable_folds, "fold_count": len(folds), "minimum_profitable_folds": 2}

    independent_decisions = np.array(["WAIT"] * len(decided240), dtype=object)
    for side, candidate in (("LONG", "long_setup"), ("SHORT", "short_setup")):
        rule_id = policy.get(side)
        if rule_id:
            independent_decisions[(decided240.candidate.to_numpy() == candidate) & funcs[rule_id](decided240).fillna(False).to_numpy()] = side
    selected = np.isin(independent_decisions, ["LONG", "SHORT"])
    independent_values = pd.to_numeric(decided240.loc[selected, "net_spread_directional_bps"], errors="coerce").dropna().to_numpy(float)
    independent_metrics = {
        "n": len(independent_values),
        "mean_net_bps": round(float(independent_values.mean()), 4) if len(independent_values) else None,
        "positive_rate": round(float((independent_values > 0).mean()), 4) if len(independent_values) else None,
        "cumulative_net_bps": round(float(independent_values.sum()), 4) if len(independent_values) else 0.0,
    }
    native = report["horizon_240m"]["full_period"]["gen11"]
    native_compare = {"n": native.get("n", 0), "mean_net_bps": native.get("mean_net_bps"), "positive_rate": native.get("positive_rate"), "cumulative_net_bps": native.get("cumulative_net_bps", 0.0)}
    report["independent_validation"] = {"implementation": "independent numpy mask evaluator", "metrics": independent_metrics, "native_metrics": native_compare, "parity": independent_metrics == native_compare}

    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    if args.policy_out:
        Path(args.policy_out).write_text(json.dumps({"version": "gen1.1-shadow-policy-v1", "research_only": True, "execution_allowed": False, "locked_policy": policy, "discovery_quantiles": thresholds, "edge_proven": report["edge_proven"]}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"edge_proven": report["edge_proven"], "policy": policy, "tested_hypotheses": len(rules), "final_240m": report["horizon_240m"]["final_holdout"], "out": args.out}, indent=2))


if __name__ == "__main__":
    main()
