"""Shared GEN1 Trade Gold orchestration core.

Both the assistant command and the PanWatch GEN1 GOLD UI call this exact
function. Pipeline order is frozen:
Ahmed ToolBox-backed macro research -> PanWatch market state -> Gen1 fusion.

Persistent memory and evidence fusion are shared with the paper/replay stack.
Every degraded/missing layer is returned explicitly. Live execution is never enabled.
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
from src.modules.xau.paper import load_gen1_memory_snapshot, record_gen1_live_observation
from src.modules.xau.service import build_decision_fusion, get_macro_context, get_xau_snapshot


def _runtime_revision() -> str:
    for key in ("RAILWAY_GIT_COMMIT_SHA","GIT_COMMIT_SHA","VERCEL_GIT_COMMIT_SHA","GITHUB_SHA"):
        value = str(os.environ.get(key) or "").strip()
        if value:
            return value[:64]
    return "unversioned-local"


async def run_gen1_trade_gold_pipeline(
    *,
    force_macro: bool = True,
    record_observation: bool = True,
    observation_source: str = "interactive",
) -> dict[str, Any]:
    stage_errors: dict[str, str] = {}

    # 1) Ahmed ToolBox-backed external/macro evidence.
    try:
        macro = await asyncio.wait_for(get_macro_context(force=force_macro), timeout=75.0)
    except Exception as exc:  # noqa: BLE001
        stage_errors["ahmed_toolbox_macro"] = type(exc).__name__
        try:
            macro = await get_macro_context(force=False)
        except Exception as fallback_exc:  # noqa: BLE001
            stage_errors["macro_fallback"] = type(fallback_exc).__name__
            macro = {
                "bias": 0, "bias_label": "neutral", "confidence": 0.0,
                "event_risk": False, "search_ok": False, "search_source": None,
                "synthesis_ok": False, "summary": "Macro layer unavailable.", "drivers": [],
            }

    # 2) PanWatch current market state.
    try:
        technical = await get_xau_snapshot(force=False)
    except Exception as exc:  # noqa: BLE001
        stage_errors["panwatch"] = type(exc).__name__
        technical = {
            "instrument": "XAUUSD", "status": "unavailable", "candidate": "none",
            "blocked": True, "block_reasons": [f"panwatch_unavailable:{type(exc).__name__}"],
            "warnings": [], "frames": {}, "execution_allowed": False,
            "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
        }

    # 2b) PanWatch persistent episodic/calibration memory.
    memory = await asyncio.to_thread(load_gen1_memory_snapshot, technical, macro)
    if not memory.get("available"):
        stage_errors["panwatch_memory"] = str(memory.get("error") or "unavailable")

    # 3) Gen1 cognition receives the same memory used by paper trading.
    try:
        fusion = build_decision_fusion(technical, macro, memory=memory)
    except Exception as exc:  # noqa: BLE001
        stage_errors["gen1"] = type(exc).__name__
        fusion = {
            "state": "unavailable", "technical_candidate": technical.get("candidate", "none"),
            "research_ready": False, "paper_entry_allowed": False, "execution_allowed": False,
            "reasons": [f"gen1_fusion_unavailable:{type(exc).__name__}"],
            "cognition": {},
        }

    # 3b) Independent-family evidence confirmation. XAUT is one bounded family,
    # not several independent votes.
    try:
        evidence = build_gen1_evidence_fusion(
            technical,
            macro,
            fusion,
            memory=memory,
            require_xaut=True,
        )
    except Exception as exc:  # noqa: BLE001
        stage_errors["evidence_fusion"] = type(exc).__name__
        evidence = {
            "version": "gen1-evidence-fusion-v1",
            "decision": "WAIT",
            "decision_confidence": fusion.get("cognitive_confidence"),
            "score": 0.0,
            "coverage": 0.0,
            "families": [],
            "conflicts": [],
            "reasons": [f"evidence_fusion_unavailable:{type(exc).__name__}"],
            "execution_allowed": False,
        }

    xaut = technical.get("xaut_order_flow") or {}
    footprint = xaut.get("footprint") or {}
    profile = xaut.get("volume_profile") or {}
    raw_book = xaut.get("raw_book") or {}
    market_context = technical.get("market_context") or {}

    missing_layers: list[str] = []
    search_source = str(macro.get("search_source") or "")
    if search_source != "ahmed_toolbox":
        missing_layers.append("ahmed_toolbox_primary_search")
    if not market_context or technical.get("market_context_error"):
        missing_layers.append("higher_timeframe_market_context")
    if not memory.get("available"):
        missing_layers.append("panwatch_persistent_memory")
    if not xaut or technical.get("xaut_order_flow_error"):
        missing_layers.append("xaut_order_flow")
    if not footprint.get("available"):
        missing_layers.append("xaut_footprint")
    if profile.get("status") != "ready":
        missing_layers.append("xaut_volume_profile")
    if not raw_book:
        missing_layers.append("xaut_raw_book")

    toolbox_stage = {
        "status": "ready" if macro.get("search_ok") and search_source == "ahmed_toolbox" else "degraded",
        "search_source": macro.get("search_source"),
        "search_ok": bool(macro.get("search_ok")),
        "synthesis_ok": bool(macro.get("synthesis_ok")),
        "summary": macro.get("summary"),
        "drivers": list(macro.get("drivers") or [])[:3],
    }
    panwatch_stage = {
        "status": technical.get("status"),
        "candidate": technical.get("candidate"),
        "technical_mode": technical.get("technical_mode"),
        "alignment": technical.get("alignment"),
        "market_context_ready": bool(market_context),
        "memory_ready": bool(memory.get("available")),
        "memory_samples": max(
            int(memory.get("similar_samples") or 0),
            int(memory.get("calibration_sample_count") or 0),
            int(memory.get("trade_count") or 0),
        ),
        "xaut_ready": bool(xaut) and not technical.get("xaut_order_flow_error"),
        "footprint_ready": bool(footprint.get("available")),
        "volume_profile_ready": profile.get("status") == "ready",
        "raw_book_ready": bool(raw_book),
    }
    gen1_stage = {
        "status": fusion.get("state"),
        "candidate": fusion.get("technical_candidate"),
        "regime": fusion.get("regime"),
        "cognitive_confidence": fusion.get("cognitive_confidence"),
        "confidence": evidence.get("decision_confidence"),
        "evidence_score": evidence.get("score"),
        "evidence_coverage": evidence.get("coverage"),
        "meta_decision": fusion.get("meta_decision"),
        "research_ready": bool(fusion.get("research_ready")),
        "paper_entry_allowed": bool(fusion.get("paper_entry_allowed")),
        "execution_allowed": False,
    }
    pipeline_status = "ready" if not missing_layers and not stage_errors else "degraded"

    result = {
        "contract": "gen1-trade-gold-v2",
        "strategy_revision": _runtime_revision(),
        "observation_source": str(observation_source or "interactive"),
        "force_macro": bool(force_macro),
        "trigger": "Gen1 trade gold",
        "pipeline_order": ["ahmed_toolbox", "panwatch", "gen1"],
        "pipeline_status": pipeline_status,
        "decision": evidence.get("decision") or "WAIT",
        "confidence": evidence.get("decision_confidence"),
        "stages": {"ahmed_toolbox": toolbox_stage, "panwatch": panwatch_stage, "gen1": gen1_stage},
        "missing_layers": missing_layers,
        "stage_errors": stage_errors,
        "macro": macro,
        "technical": technical,
        "memory": memory,
        "fusion": fusion,
        "evidence_fusion": evidence,
        "forward_range_map": xaut.get("forward_range_map"),
        "answer_contract": {
            "required": [
                "LONG_SHORT_WAIT", "confidence", "primary_scenario", "alternative_scenario",
                "invalidation", "plus_minus_10_20_30", "confirmations", "missing_layers",
            ],
            "never_hide_missing_layer": True,
            "never_treat_xaut_as_global_xauusd_order_flow": True,
            "execution_allowed": False,
        },
    }
    if record_observation:
        observation = await asyncio.to_thread(record_gen1_live_observation, result)
    else:
        observation = {"status": "skipped", "reason": "record_observation_false"}
    result["live_observation"] = observation
    return result
