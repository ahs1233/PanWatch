"""GEN1.1 research-only conditional-edge analysis for the 2Y XAU replay.

The selector is intentionally chronological and transparent:
- Discovery: first 8 months.
- Validation: next 8 months.
- Final protocol holdout: final 8 months.

Only discovery/validation are used to select rules. The final holdout is scored
only after policy lock. This module never changes the live execution path.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

try:
    import ruptures as rpt
except Exception:
    rpt = None

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except Exception:
    ColumnTransformer = permutation_importance = LogisticRegression = None
    Pipeline = OneHotEncoder = StandardScaler = None

UTC = timezone.utc
RNG_SEED = 20260923
DISCOVERY_END = datetime(2025, 5, 23, tzinfo=UTC)
VALIDATION_END = datetime(2026, 1, 23, tzinfo=UTC)
FINAL_END = datetime(2026, 9, 23, tzinfo=UTC)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _nested(mapping: dict[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = mapping or {}
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def _session(ts: datetime) -> str:
    hour = ts.astimezone(UTC).hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def _session_transition(ts: datetime) -> str:
    hour = ts.astimezone(UTC).hour
    if hour in {7, 8}:
        return "asia_london_transition"
    if hour in {12, 13}:
        return "london_new_york_transition"
    if hour in {20, 21}:
        return "new_york_off_hours_transition"
    if hour in {23, 0}:
        return "off_hours_asia_transition"
    return "none"


def _candidate_direction(candidate: str) -> str:
    return "LONG" if candidate == "long_setup" else "SHORT" if candidate == "short_setup" else "WAIT"


def _driver_map(features: dict[str, Any]) -> dict[str, float | None]:
    out = {"dxy_5d": None, "vix_5d": None, "spx_5d": None, "oil_5d": None, "us10y_5d_delta": None}
    for row in features.get("macro_drivers") or []:
        if isinstance(row, dict) and str(row.get("name") or "") in out:
            out[str(row["name"])] = _num(row.get("value"))
    return out


def flatten_episode(ep: Any) -> dict[str, Any]:
    meta = dict(getattr(ep, "meta", {}) or {})
    features = dict(meta.get("research_features") or {})
    intraday = dict(features.get("intraday") or {})
    htf = dict(features.get("htf") or {})
    market = dict(features.get("market") or {})
    state = dict(getattr(ep, "state_vector", {}) or {})
    evidence = dict(meta.get("evidence_fusion") or {})
    observed = getattr(ep, "observed_at")
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=UTC)
    observed = observed.astimezone(UTC)
    entry = float(getattr(ep, "entry_price"))
    row: dict[str, Any] = {
        "observed_at": observed,
        "horizon_minutes": int(getattr(ep, "horizon_minutes")),
        "candidate": getattr(ep, "candidate"),
        "direction": _candidate_direction(str(getattr(ep, "candidate"))),
        "regime": str(getattr(ep, "regime") or "unknown"),
        "cognitive_confidence": _num(getattr(ep, "confidence", None)),
        "gen1_decision": str(meta.get("gen1_decision") or "WAIT").upper(),
        "gen1_confidence": _num(meta.get("gen1_decision_confidence")),
        "fusion_state": str(meta.get("gen1_fusion_state") or "unknown"),
        "gross_bps": _num(getattr(ep, "directional_return_bps", None)),
        "net_bps": _num(meta.get("net_spread_directional_return_bps")),
        "session": _session(observed),
        "session_transition": _session_transition(observed),
        "hour": observed.hour,
        "day_of_week": observed.strftime("%A"),
        "month": observed.strftime("%Y-%m"),
        "technical_alignment": str(features.get("technical_alignment") or "unknown"),
        "atr_reference": _num(features.get("atr_reference")),
        "swing_high_distance_atr": _num(features.get("swing_high_distance_atr")),
        "swing_low_distance_atr": _num(features.get("swing_low_distance_atr")),
        "macro_bias": _num(features.get("macro_bias")),
        "macro_bias_label": str(features.get("macro_bias_label") or "neutral"),
        "macro_proxy_score": _num(features.get("macro_proxy_score")),
        "macro_confidence": _num(features.get("macro_confidence")),
        "htf_composite_score": _num(market.get("htf_composite_score")),
        "htf_composite_direction": str(market.get("htf_composite_direction") or "neutral"),
        "today_score": _num(market.get("today_score")),
        "today_direction": str(market.get("today_direction") or "neutral"),
        "cash_flow_score": _num(market.get("cash_flow_score")),
        "cash_flow_direction": str(market.get("cash_flow_direction") or "unknown"),
        "smart_money_score": _num(market.get("smart_money_score")),
        "smart_money_bias": str(market.get("smart_money_bias") or "unknown"),
        "volume_profile_location": str(market.get("volume_profile_location") or "unknown"),
        "dealing_zone": str(market.get("dealing_zone") or "unknown"),
        "break_of_structure": str(market.get("break_of_structure") or "none"),
        "liquidity_sweep": str(market.get("liquidity_sweep") or "none"),
        "displacement": str(market.get("displacement") or "none"),
        "rsi_5m": _num(_nested(intraday, "5m", "rsi14")),
        "rsi_15m": _num(_nested(intraday, "15m", "rsi14")),
        "atr_pct_5m": _num(_nested(intraday, "5m", "atr_pct")),
        "atr_pct_15m": _num(_nested(intraday, "15m", "atr_pct")),
        "breakout_5m": str(_nested(intraday, "5m", "breakout", default="none")),
        "breakout_15m": str(_nested(intraday, "15m", "breakout", default="none")),
        "state_trend_strength": _num(state.get("trend_strength")),
        "state_volatility": _num(state.get("volatility")),
        "state_directional_edge": _num(state.get("directional_edge")),
        "evidence_coverage": _num(evidence.get("coverage")),
        "evidence_decision_confidence": _num(evidence.get("decision_confidence")),
        **_driver_map(features),
    }
    for name in ("monthly", "weekly", "daily", "h4", "h1"):
        state_htf = dict(htf.get(name) or {})
        row[f"htf_{name}_direction"] = str(state_htf.get("direction") or "neutral")
        row[f"htf_{name}_score"] = _num(state_htf.get("score"))
        close = _num(state_htf.get("close"))
        ema = dict(state_htf.get("ema") or {})
        for period in (9, 21, 50, 200, 1000):
            value = _num(ema.get(str(period)))
            row[f"ema_{period}_{name}"] = value
            row[f"price_vs_ema_{period}_{name}_pct"] = ((entry - value) / value) * 100.0 if value not in (None, 0.0) else None
        row[f"htf_{name}_available"] = bool(state_htf.get("available"))
        row[f"htf_{name}_close"] = close
    side = "bullish" if row["direction"] == "LONG" else "bearish"
    dirs = [row[f"htf_{n}_direction"] for n in ("monthly", "weekly", "daily", "h4", "h1") if row[f"htf_{n}_available"]]
    row["htf_alignment_count"] = sum(v == side for v in dirs)
    row["htf_conflict_count"] = sum(v not in {side, "neutral", "unknown"} for v in dirs)
    row["htf_alignment_ratio"] = row["htf_alignment_count"] / len(dirs) if dirs else None
    ranges = dict(meta.get("range_outcomes") or {})
    levels = dict(ranges.get("levels") or {})
    for distance in (10, 20, 30):
        first = str((levels.get(f"pm{distance}") or {}).get("first_hit") or "none")
        row[f"pm{distance}_first_hit"] = first
        favorable = "up" if row["direction"] == "LONG" else "down"
        adverse = "down" if row["direction"] == "LONG" else "up"
        row[f"pm{distance}_favorable_first"] = first == favorable
        row[f"pm{distance}_adverse_first"] = first == adverse
    row["max_up_usd"] = _num(ranges.get("max_up_usd"))
    row["max_down_usd"] = _num(ranges.get("max_down_usd"))
    return row


def _decorrelate(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ordered = df.sort_values("observed_at")
    chosen, last = [], None
    gap = pd.Timedelta(minutes=int(minutes))
    for idx, ts in zip(ordered.index, ordered["observed_at"]):
        ts = pd.Timestamp(ts)
        if last is None or ts - last >= gap:
            chosen.append(idx)
            last = ts
    return df.loc[chosen].sort_values("observed_at").reset_index(drop=True)


def _bootstrap_mean_ci(values: np.ndarray, iterations: int = 4000) -> list[float | None]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [None, None]
    rng = np.random.default_rng(RNG_SEED)
    means = []
    for _ in range(0, iterations, 500):
        n = min(500, iterations - len(means) * 500)
        means.extend(rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1).tolist())
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _metric_block(df: pd.DataFrame, return_col: str = "net_bps") -> dict[str, Any]:
    if df.empty or return_col not in df:
        return {"n": 0}
    numeric = pd.to_numeric(df[return_col], errors="coerce")
    mask = numeric.notna()
    values = numeric[mask].to_numpy(dtype=float)
    if not len(values):
        return {"n": 0}
    wins = int(np.sum(values > 0))
    wilson = proportion_confint(wins, len(values), alpha=0.05, method="wilson")
    equity = np.cumsum(values)
    peaks = np.maximum.accumulate(np.r_[0.0, equity])[1:]
    monthly = pd.DataFrame({"month": df.loc[mask, "month"].to_numpy(), "value": values}).groupby("month")["value"].sum()
    gross = pd.to_numeric(df.loc[mask, "gross_bps"], errors="coerce") if "gross_bps" in df else pd.Series(dtype=float)
    return {
        "n": int(len(values)),
        "mean_gross_bps": round(float(gross.mean()), 4) if len(gross) else None,
        "mean_net_bps": round(float(values.mean()), 4),
        "median_net_bps": round(float(np.median(values)), 4),
        "positive_rate": round(wins / len(values), 4),
        "wilson_95_ci": [round(float(wilson[0]), 4), round(float(wilson[1]), 4)],
        "bootstrap_mean_95_ci_bps": [round(v, 4) if v is not None else None for v in _bootstrap_mean_ci(values)],
        "cumulative_net_bps": round(float(values.sum()), 4),
        "max_drawdown_bps": round(float(min(0.0, float((equity - peaks).min()))), 4),
        "profitable_month_fraction": round(float((monthly > 0).mean()), 4) if len(monthly) else None,
        "monthly_count": int(len(monthly)),
    }


def _range_block(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for d in (10, 20, 30):
        fav, adv = df[f"pm{d}_favorable_first"].astype(bool), df[f"pm{d}_adverse_first"].astype(bool)
        resolved = int((fav | adv).sum())
        out[f"pm{d}"] = {"resolved": resolved, "favorable_first": int(fav.sum()), "adverse_first": int(adv.sum()),
                         "favorable_first_rate": round(float(fav.sum() / resolved), 4) if resolved else None}
    return out


def _one_sample_permutation_p(values: np.ndarray, iterations: int = 4000) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return 1.0
    observed, rng, exceed = float(values.mean()), np.random.default_rng(RNG_SEED + len(values)), 0
    for _ in range(0, iterations, 500):
        n = min(500, iterations - _)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n, len(values)))
        exceed += int(np.sum((signs * values).mean(axis=1) >= observed))
    return float((exceed + 1) / (iterations + 1))


def _rule_id(direction: str, clauses: list[tuple[str, str, Any]]) -> str:
    return f"{direction}__" + "__".join(f"{n}:{op}:{v}" for n, op, v in clauses)


def _apply_rule(df: pd.DataFrame, rule: dict[str, Any]) -> pd.Series:
    mask = df["direction"].eq(rule["direction"])
    for column, op, value in rule["clauses"]:
        if column not in df:
            return pd.Series(False, index=df.index)
        if op == "eq":
            mask &= df[column].astype(str).eq(str(value))
        elif op == "ge":
            mask &= pd.to_numeric(df[column], errors="coerce").ge(float(value))
        elif op == "le":
            mask &= pd.to_numeric(df[column], errors="coerce").le(float(value))
        else:
            raise ValueError(op)
    return mask


def _candidate_rule_library(discovery: pd.DataFrame) -> list[dict[str, Any]]:
    rules = []
    categorical = ["session","session_transition","volatility_quartile","regime","technical_alignment",
        "htf_composite_direction","today_direction","cash_flow_direction","smart_money_bias","volume_profile_location",
        "dealing_zone","break_of_structure","liquidity_sweep","displacement","macro_bias_label",
        "htf_monthly_direction","htf_weekly_direction","htf_daily_direction","htf_h4_direction","htf_h1_direction",
        "gen1_decision","fusion_state"]
    numeric = ["cognitive_confidence","gen1_confidence","atr_pct_5m","atr_pct_15m","rsi_5m","rsi_15m",
        "htf_composite_score","today_score","cash_flow_score","smart_money_score","macro_proxy_score","htf_alignment_ratio",
        "htf_conflict_count","dxy_5d","us10y_5d_delta","vix_5d","spx_5d","oil_5d","state_trend_strength","state_volatility"]
    for direction in ("LONG", "SHORT"):
        side = discovery[discovery["direction"].eq(direction)]
        if len(side) < 60:
            continue
        for c in categorical:
            if c in side:
                for v, count in side[c].dropna().astype(str).value_counts().items():
                    if count >= 30 and v not in {"unknown","none","nan",""}:
                        cl = [(c, "eq", v)]
                        rules.append({"id": _rule_id(direction, cl), "direction": direction, "clauses": cl})
        for c in numeric:
            if c not in side:
                continue
            s = pd.to_numeric(side[c], errors="coerce").dropna()
            if len(s) >= 80 and s.nunique() >= 6:
                q25, q50, q75 = [float(s.quantile(q)) for q in (.25,.5,.75)]
                candidates = [("le", q25), ("ge", q75)] + ([("ge", q50), ("le", q50)] if c in {"rsi_5m","rsi_15m"} else [])
                for op, v in candidates:
                    cl = [(c, op, round(v, 8))]
                    rules.append({"id": _rule_id(direction, cl), "direction": direction, "clauses": cl})
        for a,b in (("session","regime"),("session","volatility_quartile"),("session","htf_daily_direction"),
                    ("regime","htf_daily_direction"),("volatility_quartile","htf_daily_direction"),
                    ("htf_weekly_direction","htf_daily_direction")):
            if a in side and b in side:
                for (va,vb), count in side.groupby([a,b]).size().sort_values(ascending=False).head(8).items():
                    if count >= 30:
                        cl=[(a,"eq",str(va)),(b,"eq",str(vb))]
                        rules.append({"id":_rule_id(direction,cl),"direction":direction,"clauses":cl})
    return list({r["id"]: r for r in rules}.values())


def _evaluate_rules(discovery: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule in _candidate_rule_library(discovery):
        dm, vm = _apply_rule(discovery, rule), _apply_rule(validation, rule)
        d, v = discovery[dm], validation[vm]
        if len(d) < 30 or len(v) < 20:
            continue
        comp = validation.loc[~vm, "net_bps"].dropna().to_numpy(dtype=float)
        try:
            mw = float(stats.mannwhitneyu(v["net_bps"].to_numpy(dtype=float), comp, alternative="greater").pvalue) if len(comp)>=10 else 1.0
        except Exception:
            mw = 1.0
        rows.append({"rule_id":rule["id"],"direction":rule["direction"],"clauses_json":json.dumps(rule["clauses"],sort_keys=True),
            "discovery_n":len(d),"discovery_mean_net_bps":float(d["net_bps"].mean()),"discovery_positive_rate":float((d["net_bps"]>0).mean()),
            "validation_n":len(v),"validation_mean_net_bps":float(v["net_bps"].mean()),"validation_positive_rate":float((v["net_bps"]>0).mean()),
            "validation_perm_p":_one_sample_permutation_p(v["net_bps"].to_numpy(dtype=float)),
            "validation_mannwhitney_p":mw if math.isfinite(mw) else 1.0,
            "stable_positive":bool(d["net_bps"].mean()>0 and v["net_bps"].mean()>0),"rule":rule})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    p = np.maximum(result["validation_perm_p"].to_numpy(), result["validation_mannwhitney_p"].to_numpy())
    result["fdr_q"] = multipletests(p, alpha=.10, method="fdr_bh")[1]
    result["score"] = result["validation_mean_net_bps"] * np.sqrt(result["validation_n"]) * np.sqrt(result["discovery_n"])
    return result.sort_values(["fdr_q","score"], ascending=[True,False]).reset_index(drop=True)


def _lock_policy(table: pd.DataFrame) -> dict[str, Any]:
    selected = []
    if not table.empty:
        eligible = table[table["stable_positive"] & (table["fdr_q"]<=.20) & (table["validation_n"]>=25) & (table["validation_positive_rate"]>=.50)]
        for d in ("LONG","SHORT"):
            pool=eligible[eligible["direction"].eq(d)]
            if not pool.empty:
                selected.append(dict(pool.sort_values(["fdr_q","score"],ascending=[True,False]).iloc[0]["rule"]))
    return {"version":"gen1.1-research-policy-v1","research_only":True,"execution_allowed":False,
            "selector":"bounded transparent subgroup library + validation FDR","fdr_method":"Benjamini-Hochberg",
            "fdr_threshold":.20,"selected_rules":selected,"default":"WAIT"}


def _apply_policy(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    out=df.copy(); out["gen11_decision"]="WAIT"; out["gen11_rule_id"]=None
    for rule in policy.get("selected_rules") or []:
        m=_apply_rule(out,rule); out.loc[m,"gen11_decision"]=rule["direction"]; out.loc[m,"gen11_rule_id"]=rule["id"]
    return out


def _period_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    ts=pd.to_datetime(df["observed_at"],utc=True)
    return {"discovery":ts<pd.Timestamp(DISCOVERY_END),
            "validation":(ts>=pd.Timestamp(DISCOVERY_END))&(ts<pd.Timestamp(VALIDATION_END)),
            "final_holdout":(ts>=pd.Timestamp(VALIDATION_END))&(ts<pd.Timestamp(FINAL_END))}


def _fold_stability(df: pd.DataFrame, decision_col: str) -> dict[str, Any]:
    active=df[df[decision_col].isin(["LONG","SHORT"])].sort_values("observed_at").reset_index(drop=True)
    if active.empty: return {"folds":[],"profitable_fold_count":0,"fold_count":0}
    items=[]
    for i,idx in enumerate(np.array_split(active.index.to_numpy(),4),1):
        items.append({"fold":i,**_metric_block(active.loc[idx])})
    return {"folds":items,"profitable_fold_count":sum((x.get("mean_net_bps") or 0)>0 for x in items),"fold_count":len(items)}


def _subgroup_table(df: pd.DataFrame) -> pd.DataFrame:
    records=[]; masks=_period_masks(df)
    dims=["direction","session","session_transition","volatility_quartile","regime","gen1_decision","fusion_state","technical_alignment",
          "htf_monthly_direction","htf_weekly_direction","htf_daily_direction","htf_h4_direction","htf_h1_direction",
          "htf_composite_direction","today_direction","cash_flow_direction","smart_money_bias","volume_profile_location","day_of_week","hour"]
    for dim in dims:
        if dim not in df: continue
        for value,group in df.groupby(dim,dropna=False):
            if len(group)<20: continue
            row={"dimension":dim,"value":str(value)}
            for period,mask in masks.items():
                metrics=_metric_block(group[group.index.isin(df.index[mask])])
                for k,v in metrics.items():
                    if k=="n" or isinstance(v,(int,float,str)) or v is None: row[f"{period}_{k}"]=v
                ci=metrics.get("bootstrap_mean_95_ci_bps") or [None,None]; wil=metrics.get("wilson_95_ci") or [None,None]
                row[f"{period}_ci_low"],row[f"{period}_ci_high"]=ci; row[f"{period}_wilson_low"],row[f"{period}_wilson_high"]=wil
            records.append(row)
    return pd.DataFrame(records)


def _feature_importance(discovery: pd.DataFrame, validation: pd.DataFrame) -> list[dict[str, Any]]:
    if Pipeline is None or discovery.empty or validation.empty: return []
    numeric=[c for c in ["cognitive_confidence","gen1_confidence","atr_pct_5m","atr_pct_15m","rsi_5m","rsi_15m","htf_composite_score",
      "today_score","cash_flow_score","smart_money_score","macro_proxy_score","htf_alignment_ratio","htf_conflict_count","dxy_5d",
      "us10y_5d_delta","vix_5d","spx_5d","oil_5d","state_trend_strength","state_volatility"] if c in discovery]
    categorical=[c for c in ["direction","session","session_transition","volatility_quartile","regime","technical_alignment","htf_monthly_direction",
      "htf_weekly_direction","htf_daily_direction","htf_h4_direction","htf_h1_direction","htf_composite_direction","cash_flow_direction",
      "smart_money_bias","volume_profile_location","gen1_decision","fusion_state"] if c in discovery]
    xt=discovery[numeric+categorical].copy(); xv=validation[numeric+categorical].copy()
    for c in numeric:
        med=pd.to_numeric(xt[c],errors="coerce").median(); med=0.0 if pd.isna(med) else float(med)
        xt[c]=pd.to_numeric(xt[c],errors="coerce").fillna(med); xv[c]=pd.to_numeric(xv[c],errors="coerce").fillna(med)
    for c in categorical: xt[c]=xt[c].fillna("unknown").astype(str); xv[c]=xv[c].fillna("unknown").astype(str)
    yt=(discovery["net_bps"]>0).astype(int); yv=(validation["net_bps"]>0).astype(int)
    if yt.nunique()<2 or yv.nunique()<2: return []
    trans=[]
    if numeric: trans.append(("num",StandardScaler(),numeric))
    if categorical: trans.append(("cat",OneHotEncoder(handle_unknown="ignore"),categorical))
    model=Pipeline([("pre",ColumnTransformer(trans)),("clf",LogisticRegression(max_iter=1000,penalty="l1",solver="liblinear",C=.5,random_state=RNG_SEED))])
    model.fit(xt,yt)
    pi=permutation_importance(model,xv,yv,n_repeats=10,random_state=RNG_SEED,scoring="roc_auc")
    return [{"feature":n,"permutation_importance_auc":round(float(m),6),"std":round(float(s),6)}
            for n,m,s in sorted(zip(numeric+categorical,pi.importances_mean,pi.importances_std),key=lambda x:x[1],reverse=True)[:15]]


def _robust_univariate(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows=[]
    for c in ["cognitive_confidence","gen1_confidence","atr_pct_5m","atr_pct_15m","rsi_5m","rsi_15m","htf_composite_score","today_score",
              "cash_flow_score","smart_money_score","macro_proxy_score","htf_alignment_ratio","state_trend_strength","state_volatility","state_directional_edge"]:
        if c not in df: continue
        x=pd.to_numeric(df[c],errors="coerce"); y=pd.to_numeric(df["net_bps"],errors="coerce"); valid=x.notna()&y.notna()
        if valid.sum()<50 or x[valid].nunique()<5: continue
        rho,p=stats.spearmanr(x[valid],y[valid]); rows.append({"feature":c,"spearman_rho":float(rho),"p":float(p),"n":int(valid.sum())})
    if rows:
        for row,q in zip(rows,multipletests([r["p"] for r in rows],alpha=.10,method="fdr_bh")[1]): row["fdr_q"]=float(q)
    return sorted(rows,key=lambda r:(r.get("fdr_q",1.0),-abs(r["spearman_rho"])))


def _robust_ols(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows=[]; yall=pd.to_numeric(df["net_bps"],errors="coerce")
    for c in ["cognitive_confidence","gen1_confidence","atr_pct_5m","atr_pct_15m","rsi_5m","rsi_15m","htf_composite_score","today_score",
      "cash_flow_score","smart_money_score","macro_proxy_score","htf_alignment_ratio","htf_conflict_count","state_trend_strength",
      "state_volatility","state_directional_edge","dxy_5d","us10y_5d_delta","vix_5d","spx_5d","oil_5d"]:
        if c not in df: continue
        xall=pd.to_numeric(df[c],errors="coerce"); valid=xall.notna()&yall.notna()
        if valid.sum()<60 or xall[valid].nunique()<6: continue
        x=xall[valid].astype(float); y=yall[valid].astype(float); sd=float(x.std(ddof=0))
        if not math.isfinite(sd) or sd<=1e-12: continue
        model=sm.OLS(y,sm.add_constant((x-float(x.mean()))/sd)).fit(cov_type="HC3")
        rows.append({"feature":c,"n":int(valid.sum()),"coef_bps_per_1sd":float(model.params.iloc[1]),
                     "robust_se":float(model.bse.iloc[1]),"p":float(model.pvalues.iloc[1]),"r2":float(model.rsquared)})
    if rows:
        for row,q in zip(rows,multipletests([r["p"] for r in rows],alpha=.10,method="fdr_bh")[1]): row["fdr_q"]=float(q)
    return sorted(rows,key=lambda r:(r.get("fdr_q",1.0),-abs(r["coef_bps_per_1sd"])))


def _regime_discovery(df: pd.DataFrame) -> dict[str, Any]:
    if rpt is None or len(df)<200: return {"available":False,"reason":"ruptures_unavailable_or_sample_too_small"}
    cols=[c for c in ["htf_composite_score","atr_pct_15m","cash_flow_score","smart_money_score"] if c in df]
    if len(cols)<2: return {"available":False,"reason":"insufficient_predecision_features"}
    ordered=df.sort_values("observed_at").copy(); matrix=ordered[cols].apply(pd.to_numeric,errors="coerce")
    matrix=matrix.fillna(matrix.median()).fillna(0.0); std=matrix.std().replace(0.0,1.0)
    z=((matrix-matrix.mean())/std).to_numpy(dtype=float)
    try:
        breaks=rpt.Pelt(model="rbf",min_size=40,jump=5).fit(z).predict(pen=3.0*math.log(max(2,len(z)))*z.shape[1])
    except Exception as exc:
        return {"available":False,"reason":f"ruptures_failed:{type(exc).__name__}"}
    segments=[]; start=0
    for i,stop in enumerate(breaks,1):
        seg=ordered.iloc[start:stop]
        if not seg.empty: segments.append({"segment":i,"start":pd.Timestamp(seg.iloc[0]["observed_at"]).isoformat(),
            "end":pd.Timestamp(seg.iloc[-1]["observed_at"]).isoformat(),**_metric_block(seg)})
        start=stop
    return {"available":True,"method":"ruptures.Pelt(rbf)","input_features":cols,"segment_count":len(segments),
            "segments":segments,"production_use":False,"note":"Diagnostic only; not used by Gen1.1 policy selection."}


def _independent_validation(df: pd.DataFrame, policy: dict[str, Any]) -> dict[str, Any]:
    selected=[]
    for _,row in df.iterrows():
        decision="WAIT"
        for rule in policy.get("selected_rules") or []:
            if row.get("direction")!=rule.get("direction"): continue
            matched=True
            for col,op,val in rule.get("clauses") or []:
                actual=row.get(col)
                if op=="eq": matched &= str(actual)==str(val)
                elif op=="ge": matched &= _num(actual) is not None and float(actual)>=float(val)
                elif op=="le": matched &= _num(actual) is not None and float(actual)<=float(val)
                if not matched: break
            if matched: decision=rule["direction"]; break
        if decision in {"LONG","SHORT"}: selected.append(float(row["net_bps"]))
    vec=_apply_policy(df,policy); v=vec.loc[vec["gen11_decision"].isin(["LONG","SHORT"]),"net_bps"].astype(float).to_numpy()
    l=np.asarray(selected,dtype=float); parity=len(v)==len(l) and np.allclose(v,l,atol=1e-12,rtol=0.0)
    return {"implementation":"independent_row_loop","vectorized_count":int(len(v)),"independent_count":int(len(l)),"parity":bool(parity),
            "mean_delta_bps":round(float(v.mean()-l.mean()),12) if len(v) and len(l) else 0.0}


def analyze_gen11(episodes60: list[Any], episodes240: list[Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    rows240=pd.DataFrame([flatten_episode(ep) for ep in episodes240]); rows60=pd.DataFrame([flatten_episode(ep) for ep in episodes60])
    for x in (rows240,rows60): x["observed_at"]=pd.to_datetime(x["observed_at"],utc=True)
    independent240=_decorrelate(rows240,240); independent60=_decorrelate(rows60,60)
    masks=_period_masks(independent240); discovery=independent240[masks["discovery"]].copy()
    validation=independent240[masks["validation"]].copy(); final_holdout=independent240[masks["final_holdout"]].copy()
    atr=pd.to_numeric(discovery["atr_pct_15m"],errors="coerce").dropna()
    if len(atr)>=20:
        q1,q2,q3=[float(atr.quantile(q)) for q in (.25,.5,.75)]
        def addq(frame):
            frame["volatility_quartile"]=pd.cut(pd.to_numeric(frame["atr_pct_15m"],errors="coerce"),
                bins=[-np.inf,q1,q2,q3,np.inf],labels=["q1_low","q2","q3","q4_high"],include_lowest=True).astype(str)
    else:
        def addq(frame): frame["volatility_quartile"]="unknown"
    for frame in (independent240,independent60,discovery,validation,final_holdout): addq(frame)
    table=_evaluate_rules(discovery,validation); policy=_lock_policy(table)
    policy.update({"locked_at_boundary":VALIDATION_END.isoformat(),"holdout_used_for_selection":False,
                   "discovery_window":["2024-09-23T00:00:00+00:00",DISCOVERY_END.isoformat()],
                   "validation_window":[DISCOVERY_END.isoformat(),VALIDATION_END.isoformat()],
                   "final_holdout_window":[VALIDATION_END.isoformat(),FINAL_END.isoformat()]})
    g240=_apply_policy(independent240,policy); g60=_apply_policy(independent60,policy)
    def comparison(frame):
        out={}
        for name,mask in _period_masks(frame).items():
            p=frame[mask]; gen1=p[p["gen1_decision"].isin(["LONG","SHORT"])]; gen11=p[p["gen11_decision"].isin(["LONG","SHORT"])]
            out[name]={"gen1":_metric_block(gen1),"gen1_1":_metric_block(gen11),"gen1_1_range_outcomes":_range_block(gen11)}
        return out
    comp240,comp60=comparison(g240),comparison(g60)
    final=comp240["final_holdout"]["gen1_1"]; ci=final.get("bootstrap_mean_95_ci_bps") or [None,None]
    stability=_fold_stability(g240[_period_masks(g240)["final_holdout"]],"gen11_decision")
    confirmation=bool((final.get("n") or 0)>=60 and (final.get("mean_net_bps") or 0)>0 and ci[0] is not None and ci[0]>0
                      and stability.get("profitable_fold_count",0)>=3)
    subgroup=_subgroup_table(independent240); importance=_feature_importance(discovery,validation)
    univ=_robust_univariate(pd.concat([discovery,validation],ignore_index=True))
    ols=_robust_ols(pd.concat([discovery,validation],ignore_index=True))
    regimes=_regime_discovery(pd.concat([discovery,validation],ignore_index=True)); parity=_independent_validation(final_holdout,policy)
    cols=["observed_at","direction","session","session_transition","volatility_quartile","regime","gen1_decision","fusion_state","net_bps",
      "gross_bps","cognitive_confidence","gen1_confidence","technical_alignment","atr_pct_5m","atr_pct_15m","rsi_5m","rsi_15m",
      "htf_monthly_direction","htf_weekly_direction","htf_daily_direction","htf_h4_direction","htf_h1_direction","htf_composite_score",
      "cash_flow_score","smart_money_score","macro_proxy_score","pm10_first_hit","pm20_first_hit","pm30_first_hit"]
    independent240[[c for c in cols if c in independent240]].to_csv(out_dir/"episodes_240m_enriched_independent.csv",index=False)
    if not subgroup.empty: subgroup.to_csv(out_dir/"gen1_1_subgroups.csv",index=False)
    if not table.empty: table.drop(columns=["rule"]).to_csv(out_dir/"gen1_1_rule_search.csv",index=False)
    pd.DataFrame(importance).to_csv(out_dir/"gen1_1_feature_importance.csv",index=False)
    pd.DataFrame(ols).to_csv(out_dir/"gen1_1_robust_ols.csv",index=False)
    (out_dir/"gen1_1_policy.json").write_text(json.dumps(policy,indent=2,sort_keys=True),encoding="utf-8")
    report={"version":"gen1.1-edge-analysis-v1","research_only":True,"execution_allowed":False,
      "protocol":{"discovery_end":DISCOVERY_END.isoformat(),"validation_end":VALIDATION_END.isoformat(),"final_holdout_end":FINAL_END.isoformat(),
        "selector_reads_final_holdout":False,"multiple_testing":"Benjamini-Hochberg FDR",
        "rule_library":"bounded categorical/quantile rules; no exhaustive grid search",
        "final_holdout_note":"Selector-blinded in code; earlier aggregate second-year Gen1 metrics were known, so forward paper validation remains required."},
      "counts":{"raw_240m":len(rows240),"independent_240m":len(independent240),"discovery_240m":len(discovery),
                "validation_240m":len(validation),"final_holdout_240m":len(final_holdout),"independent_60m":len(independent60)},
      "policy":policy,"comparison":{"60m":comp60,"240m":comp240},"feature_importance":importance,"univariate_fdr":univ[:20],
      "robust_ols_hc3_fdr":ols[:20],"regime_discovery":regimes,"independent_validation":parity,
      "final_holdout_fold_stability":stability,"historical_statistical_confirmation_candidate":confirmation,"edge_proven":False,
      "edge_proven_reason":"Backtest can nominate a statistical candidate; production edge stays false until forward paper OOS confirmation.",
      "selected_rule_count":len(policy.get("selected_rules") or []),
      "limitations":["Historical XAUT raw book/trade tape unavailable and not fabricated.","Macro uses lagged public market proxies.",
        "Dukascopy volume is an OTC activity proxy, not centralized global gold volume.","Missing EMA1000 is never backfilled from future data.",
        "No Gen1.1 rule is wired into live execution."]}
    (out_dir/"gen1_1_edge_report.json").write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    return report
