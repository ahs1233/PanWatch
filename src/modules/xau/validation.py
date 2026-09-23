"""Validation summaries for GEN1 Gold historical replay.

Historical validation is intentionally conservative:
- XAUT raw order book / executed-trade microstructure is unavailable historically
  unless a dedicated historical source is supplied.
- Replay outcomes are research labels, not broker fills.
- Metrics can identify an exploratory edge candidate; they never prove an edge.
"""
from __future__ import annotations

from typing import Any, Iterable

from src.modules.xau.paper_store import (
    open_xau_paper_session,
    open_xau_replay_session,
    replay_store_is_external,
)
from src.platform.persistence.models import XAUPaperSignal, XAUReplayEpisode


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
            payload["meta"]=dict(signal.meta or {})
            episodes.append(payload)
        return evaluate_gen1_replay(episodes)
    finally:
        db.close()
