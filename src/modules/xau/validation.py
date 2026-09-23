"""Validation summaries for GEN1 Gold historical replay.

Historical validation is intentionally conservative:
- XAUT raw order book / executed-trade microstructure is unavailable historically
  unless a dedicated historical source is supplied.
- Replay outcomes are research labels, not broker fills.
- Metrics can identify an exploratory edge candidate; they never prove an edge.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from src.modules.xau.paper_store import (
    open_xau_paper_session,
    open_xau_replay_session,
    replay_store_is_external,
)
from src.platform.persistence.models import XAUPaperSignal, XAUReplayEpisode
from src.platform.runtime.config import Settings


def _meta(item: Any) -> dict[str, Any]:
    value = getattr(item, "meta", None)
    if isinstance(value, dict):
        return value
    if isinstance(item, dict):
        return dict(item.get("meta") or {})
    return {}


def _number(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _calibration(predictions: list[tuple[float, int]]) -> dict[str, Any]:
    if not predictions:
        return {"sample_count": 0, "brier_score": None, "expected_calibration_error": None}
    clean = [(max(0.0, min(1.0, p)), 1 if y else 0) for p, y in predictions]
    brier = sum((p-y)**2 for p,y in clean) / len(clean)
    bins=[[] for _ in range(5)]
    for p,y in clean:
        bins[min(4,int(p*5))].append((p,y))
    ece=0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p=sum(p for p,_ in bucket)/len(bucket)
        avg_y=sum(y for _,y in bucket)/len(bucket)
        ece += len(bucket)/len(clean)*abs(avg_p-avg_y)
    return {
        "sample_count": len(clean),
        "brier_score": round(brier,4),
        "expected_calibration_error": round(ece,4),
    }


def evaluate_gen1_replay(episodes: Iterable[Any]) -> dict[str, Any]:
    rows=list(episodes)
    decisions={"LONG":0,"SHORT":0,"WAIT":0,"UNKNOWN":0}
    directional=[]
    baseline=[]
    predictions=[]
    ranges={10:{"favorable_first":0,"adverse_first":0,"ambiguous":0,"neither":0},
            20:{"favorable_first":0,"adverse_first":0,"ambiguous":0,"neither":0},
            30:{"favorable_first":0,"adverse_first":0,"ambiguous":0,"neither":0}}
    sensor_gaps=set()

    for item in rows:
        meta=_meta(item)
        decision=str(meta.get("gen1_decision") or "UNKNOWN").upper()
        if decision not in decisions:
            decision="UNKNOWN"
        decisions[decision]+=1

        candidate=str(getattr(item,"candidate", "") or (item.get("candidate") if isinstance(item,dict) else ""))
        dr=_number(getattr(item,"directional_return_bps",None))
        if dr is None and isinstance(item,dict):
            dr=_number(item.get("directional_return_bps"))
        if candidate in {"long_setup","short_setup"} and dr is not None:
            baseline.append(float(dr))

        for gap in meta.get("sensor_gaps") or []:
            sensor_gaps.add(str(gap))

        if decision not in {"LONG","SHORT"}:
            continue
        if dr is None:
            continue
        directional.append(float(dr))
        conf=_number(meta.get("gen1_decision_confidence"))
        if conf is not None:
            predictions.append((float(conf),1 if dr>0 else 0))

        range_payload=(meta.get("range_outcomes") or {}).get("levels") or {}
        for distance in (10,20,30):
            payload=range_payload.get(f"pm{distance}") or {}
            first=str(payload.get("first_hit") or "none")
            favorable="up" if decision=="LONG" else "down"
            adverse="down" if decision=="LONG" else "up"
            bucket=ranges[distance]
            if first==favorable:
                bucket["favorable_first"]+=1
            elif first==adverse:
                bucket["adverse_first"]+=1
            elif first=="ambiguous_same_bar":
                bucket["ambiguous"]+=1
            else:
                bucket["neither"]+=1

    directional_count=len(directional)
    positive=sum(1 for value in directional if value>0)
    baseline_positive=sum(1 for value in baseline if value>0)
    positive_rate=positive/directional_count if directional_count else None
    baseline_rate=baseline_positive/len(baseline) if baseline else None
    avg_bps=sum(directional)/directional_count if directional_count else None
    baseline_avg=sum(baseline)/len(baseline) if baseline else None

    range_summary={}
    for distance,bucket in ranges.items():
        resolved=bucket["favorable_first"]+bucket["adverse_first"]
        range_summary[f"pm{distance}"]={
            **bucket,
            "resolved_first_touch_count": resolved,
            "favorable_first_rate": round(bucket["favorable_first"]/resolved,4) if resolved else None,
        }

    calibration=_calibration(predictions)
    enough=directional_count>=100
    exploratory=bool(
        enough
        and positive_rate is not None and positive_rate>=0.55
        and avg_bps is not None and avg_bps>0
        and calibration["brier_score"] is not None and calibration["brier_score"]<0.25
    )
    status=(
        "exploratory_edge_candidate_requires_out_of_sample_confirmation"
        if exploratory
        else "insufficient_directional_sample"
        if not enough
        else "no_exploratory_edge_detected"
    )

    return {
        "version":"gen1-gold-validation-v1",
        "episode_count":len(rows),
        "decision_counts":decisions,
        "directional_decision_count":directional_count,
        "directional_positive_rate":round(positive_rate,4) if positive_rate is not None else None,
        "average_directional_return_bps":round(avg_bps,4) if avg_bps is not None else None,
        "candidate_baseline_count":len(baseline),
        "candidate_baseline_positive_rate":round(baseline_rate,4) if baseline_rate is not None else None,
        "candidate_baseline_average_return_bps":round(baseline_avg,4) if baseline_avg is not None else None,
        "positive_rate_uplift":round(positive_rate-baseline_rate,4) if positive_rate is not None and baseline_rate is not None else None,
        "calibration":calibration,
        "range_outcomes":range_summary,
        "validation_status":status,
        "exploratory_edge_candidate":exploratory,
        "edge_proven":False,
        "sensor_gaps":sorted(sensor_gaps),
        "historical_xaut_microstructure_validated":False,
        "limitations":[
            "Historical XAUT raw-book and executed-trade microstructure are not replayed by this dataset.",
            "Replay uses research OHLC and point-in-time macro only when a historical macro provider is supplied.",
            "Replay outcomes are labels from future research bars, not executable broker fills.",
            "An exploratory positive result is not proof of a persistent trading edge.",
        ],
        "execution_allowed":False,
    }


def load_gen1_validation_summary(*, limit:int=2000)->dict[str,Any]:
    limit=max(10,min(int(limit),10000))
    if replay_store_is_external():
        db=open_xau_replay_session()
        try:
            rows=(db.query(XAUReplayEpisode)
                  .order_by(XAUReplayEpisode.observed_at.desc(),XAUReplayEpisode.id.desc())
                  .limit(limit).all())
            return evaluate_gen1_replay(rows)
        finally:
            db.close()

    db=open_xau_paper_session()
    try:
        signals=(db.query(XAUPaperSignal)
                 .filter(XAUPaperSignal.rejection_reason=="historical_replay")
                 .order_by(XAUPaperSignal.observed_at.desc(),XAUPaperSignal.id.desc())
                 .limit(limit).all())
        episodes=[]
        for signal in signals:
            payload=dict((signal.meta or {}).get("replay_episode") or {})
            if not payload:
                continue
            episode_meta=dict(payload.get("meta") or {})
            episode_meta["storage_mode"]="paper_signal_compat"
            payload["meta"]=episode_meta
            episodes.append(payload)
        return evaluate_gen1_replay(episodes)
    finally:
        db.close()



def _observed_at(item: Any) -> datetime | None:
    value = getattr(item, "observed_at", None)
    if value is None and isinstance(item, dict):
        value = item.get("observed_at")
    if value is None:
        value = _meta(item).get("observed_at")
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = p + z2 / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) / total) + z2 / (4.0 * total * total))
    return max(0.0, (centre - margin) / denominator)


def _bootstrap_mean_lower(
    values: list[float],
    *,
    iterations: int = 2000,
    seed: int = 20260923,
) -> float | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(max(200, int(iterations))):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    index = max(0, min(len(means) - 1, int(0.025 * (len(means) - 1))))
    return means[index]


def _session_name_utc(observed: datetime) -> str:
    hour = observed.astimezone(timezone.utc).hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def _live_completed_rows(signals: Iterable[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in signals:
        meta = _meta(item)
        decision = str(meta.get("decision") or "UNKNOWN").upper()
        if decision not in {"LONG", "SHORT"}:
            continue
        horizon = (meta.get("horizon_outcomes") or {}).get("60m") or {}
        directional = _number(horizon.get("directional_return_bps"))
        observed = _observed_at(item)
        if directional is None or observed is None:
            continue
        missing_layers = list(meta.get("missing_layers") or [])
        if missing_layers:
            continue
        rows.append({
            "observed_at": observed,
            "decision": decision,
            "directional_return_bps": float(directional),
            "confidence": _number(meta.get("decision_confidence")),
            "strategy_revision": str(meta.get("strategy_revision") or "unversioned-local"),
            "revision_pinning_available": bool(meta.get("revision_pinning_available")),
            "regime": str(meta.get("regime") or "unknown"),
            "observation_source": str(meta.get("observation_source") or "unknown"),
            "session": _session_name_utc(observed),
        })
    rows.sort(key=lambda row: row["observed_at"])
    return rows


def _decorrelate_live_rows(rows: list[dict[str, Any]], *, min_separation_minutes: int) -> list[dict[str, Any]]:
    chosen = []
    last_observed = None
    minimum = timedelta(minutes=max(1, int(min_separation_minutes)))
    for row in rows:
        observed = row["observed_at"]
        if last_observed is None or observed - last_observed >= minimum:
            chosen.append(row)
            last_observed = observed
    return chosen


def _forward_oos_summary(
    signals: Iterable[Any],
    *,
    min_separation_minutes: int = 15,
    min_total: int = 150,
    min_holdout: int = 50,
    holdout_fraction: float = 0.33,
) -> dict[str, Any]:
    completed = _live_completed_rows(signals)
    revision = completed[-1]["strategy_revision"] if completed else None
    same_revision = [row for row in completed if row["strategy_revision"] == revision]
    decorrelated = _decorrelate_live_rows(
        same_revision,
        min_separation_minutes=min_separation_minutes,
    )
    total = len(decorrelated)
    holdout_fraction = max(0.20, min(0.50, float(holdout_fraction)))
    target_holdout = max(int(min_holdout), int(math.ceil(total * holdout_fraction)))
    holdout_n = min(total, target_holdout)
    reference_n = max(0, total - holdout_n)
    holdout = decorrelated[-holdout_n:] if holdout_n else []

    values = [row["directional_return_bps"] for row in holdout]
    positive = sum(1 for value in values if value > 0)
    positive_rate = positive / len(values) if values else None
    wilson = _wilson_lower_bound(positive, len(values))
    average = sum(values) / len(values) if values else None
    bootstrap_lower = _bootstrap_mean_lower(values) if values else None
    predictions = [
        (float(row["confidence"]), 1 if row["directional_return_bps"] > 0 else 0)
        for row in holdout if row.get("confidence") is not None
    ]
    calibration = _calibration(predictions)
    revision_pinned = bool(
        revision and revision != "unversioned-local" and holdout
        and all(row.get("revision_pinning_available") for row in holdout)
    )

    enough_total = total >= int(min_total)
    enough_holdout = len(holdout) >= int(min_holdout) and reference_n > 0
    win_gate = wilson is not None and wilson > 0.50
    return_gate = bootstrap_lower is not None and bootstrap_lower > 0.0
    calibration_gate = (
        calibration.get("sample_count", 0) >= int(min_holdout)
        and calibration.get("brier_score") is not None
        and calibration["brier_score"] < 0.25
    )
    passed = bool(
        enough_total and enough_holdout and revision_pinned
        and win_gate and return_gate and calibration_gate
    )
    if not enough_total or not enough_holdout:
        status = "collecting_forward_oos_sample"
    elif not revision_pinned:
        status = "revision_not_pinned"
    elif passed:
        status = "oos_statistical_confirmation_candidate"
    else:
        status = "oos_gates_not_passed"

    sessions = {}
    regimes = {}
    for row in holdout:
        sessions[row["session"]] = sessions.get(row["session"], 0) + 1
        regimes[row["regime"]] = regimes.get(row["regime"], 0) + 1

    return {
        "protocol": "prequential_forward_oos_v1",
        "status": status,
        "passed": passed,
        "edge_proven": False,
        "strategy_revision": revision,
        "revision_pinning_available": revision_pinned,
        "raw_completed_current_revision": len(same_revision),
        "decorrelated_completed_current_revision": total,
        "min_separation_minutes": int(min_separation_minutes),
        "minimum_total_required": int(min_total),
        "minimum_holdout_required": int(min_holdout),
        "holdout_fraction": round(holdout_fraction, 4),
        "reference_window_count": reference_n,
        "holdout_count": len(holdout),
        "progress_fraction": round(min(1.0, total / max(1, int(min_total))), 4),
        "positive_rate": round(positive_rate, 4) if positive_rate is not None else None,
        "wilson_95_lower": round(wilson, 4) if wilson is not None else None,
        "average_directional_return_bps": round(average, 4) if average is not None else None,
        "bootstrap_mean_bps_95_lower": round(bootstrap_lower, 4) if bootstrap_lower is not None else None,
        "calibration": calibration,
        "gates": {
            "minimum_total": enough_total,
            "minimum_holdout": enough_holdout,
            "revision_pinned": revision_pinned,
            "win_rate_wilson_lower_above_50pct": win_gate,
            "mean_return_bootstrap_lower_above_zero": return_gate,
            "brier_below_0_25": calibration_gate,
        },
        "session_counts": sessions,
        "regime_counts": regimes,
        "limitations": [
            "This is forward prequential OOS evaluation: each outcome was unknown when its decision was made.",
            "Observations are decorrelated by time but can still share market-regime exposure.",
            "Outcome prices are sampled research references, not executable broker fills.",
            "Transaction costs, slippage and broker-specific fill quality are not yet part of this OOS gate.",
            "Passing these gates is statistical confirmation under this research protocol, not proof of a persistent executable edge.",
        ],
    }


def evaluate_gen1_live_observations(
    signals: Iterable[Any],
    *,
    min_separation_minutes: int = 15,
    min_oos_total: int = 150,
    min_holdout: int = 50,
    holdout_fraction: float = 0.33,
) -> dict[str, Any]:
    rows=list(signals)
    decisions={"LONG":0,"SHORT":0,"WAIT":0,"UNKNOWN":0}
    completed=[]
    predictions=[]
    ranges={10:{"favorable_first":0,"adverse_first":0,"unresolved":0},
            20:{"favorable_first":0,"adverse_first":0,"unresolved":0},
            30:{"favorable_first":0,"adverse_first":0,"unresolved":0}}
    for row in rows:
        meta=_meta(row)
        decision=str(meta.get("decision") or "UNKNOWN").upper()
        if decision not in decisions:
            decision="UNKNOWN"
        decisions[decision]+=1
        if decision in {"LONG","SHORT"}:
            horizon=(meta.get("horizon_outcomes") or {}).get("60m") or {}
            directional=_number(horizon.get("directional_return_bps"))
            confidence=_number(meta.get("decision_confidence"))
            if directional is not None:
                completed.append(float(directional))
                if confidence is not None:
                    predictions.append((float(confidence),1 if directional>0 else 0))
            levels=(meta.get("live_range_outcomes") or {}).get("levels") or {}
            for distance in (10,20,30):
                first=str((levels.get(f"pm{distance}") or {}).get("first_hit") or "none")
                favorable="up" if decision=="LONG" else "down"
                adverse="down" if decision=="LONG" else "up"
                if first==favorable:
                    ranges[distance]["favorable_first"]+=1
                elif first==adverse:
                    ranges[distance]["adverse_first"]+=1
                else:
                    ranges[distance]["unresolved"]+=1

    positive=sum(1 for value in completed if value>0)
    rate=positive/len(completed) if completed else None
    avg=sum(completed)/len(completed) if completed else None
    calibration=_calibration(predictions)
    range_summary={}
    for distance,bucket in ranges.items():
        resolved=bucket["favorable_first"]+bucket["adverse_first"]
        range_summary[f"pm{distance}"]={
            **bucket,
            "resolved_first_touch_count":resolved,
            "favorable_first_rate":round(bucket["favorable_first"]/resolved,4) if resolved else None,
            "method":"sampled_reference_first_touch",
            "exact_intrabar_order_known":False,
        }

    oos=_forward_oos_summary(
        rows,
        min_separation_minutes=min_separation_minutes,
        min_total=min_oos_total,
        min_holdout=min_holdout,
        holdout_fraction=holdout_fraction,
    )
    enough=len(completed)>=100
    exploratory=bool(
        enough and rate is not None and rate>=0.55 and avg is not None and avg>0
        and calibration["brier_score"] is not None and calibration["brier_score"]<0.25
    )
    return {
        "version":"gen1-live-validation-v2",
        "observation_count":len(rows),
        "decision_counts":decisions,
        "completed_60m_directional_count":len(completed),
        "directional_positive_rate_60m":round(rate,4) if rate is not None else None,
        "average_directional_return_bps_60m":round(avg,4) if avg is not None else None,
        "calibration":calibration,
        "range_outcomes":range_summary,
        "forward_oos":oos,
        "validation_status":oos["status"],
        "exploratory_edge_candidate":exploratory,
        "edge_proven":False,
        "full_live_pipeline_including_xaut":True,
        "first_touch_precision":"sampled_not_intrabar_exact",
        "limitations":[
            "First-touch ordering is based on scheduler observations, not tick-perfect intrabar reconstruction.",
            "60m outcomes use the first observed reference price at or after the horizon.",
            "OOS statistics use time-decorrelated observations from the latest strategy revision only.",
            "An exploratory or OOS-positive result is not proof of a persistent executable trading edge.",
        ],
        "execution_allowed":False,
    }


def load_gen1_live_validation_summary(*, limit:int=2000)->dict[str,Any]:
    limit=max(10,min(int(limit),10000))
    db=open_xau_paper_session()
    try:
        rows=(db.query(XAUPaperSignal)
              .filter(XAUPaperSignal.rejection_reason=="gen1_live_observation")
              .order_by(XAUPaperSignal.observed_at.desc(),XAUPaperSignal.id.desc())
              .limit(limit).all())
        settings=Settings()
        return evaluate_gen1_live_observations(
            rows,
            min_separation_minutes=settings.xau_gen1_oos_min_separation_minutes,
            min_oos_total=settings.xau_gen1_oos_min_total,
            min_holdout=settings.xau_gen1_oos_min_holdout,
            holdout_fraction=settings.xau_gen1_oos_holdout_fraction,
        )
    finally:
        db.close()
