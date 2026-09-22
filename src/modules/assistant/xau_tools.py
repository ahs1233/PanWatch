"""Research-only XAUUSD tools for the PanWatch assistant."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from pan_agent import (
    RunRequest,
    ToolExposure,
    ToolRegistry,
    ToolResult,
    ToolRisk,
    ToolSpec,
)
from pan_agent_tool_research import (
    ToolDataFreshness,
    ToolDescriptor,
)

from src.modules.xau.paper import XAUPaperTradingEngine
from src.modules.xau.service import (
    build_decision_fusion,
    get_macro_context,
    get_xau_snapshot,
)


def register_xau_research_tools(registry: ToolRegistry) -> list[ToolDescriptor]:
    """Register the same XAU research state used by the terminal UI."""

    async def get_xau_intraday_research(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        try:
            snapshot = await get_xau_snapshot(force=False)
        except Exception as exc:  # noqa: BLE001 - surface provider failure as a tool result
            return ToolResult.failure(
                summary=f"XAU intraday research unavailable: {type(exc).__name__}",
                error_code="xau_research_unavailable",
            )

        spot = snapshot.get("indicative_spot") or {}
        micro = snapshot.get("micro") or {}
        summary = (
            "XAUUSD intraday research: "
            f"status={snapshot.get('status')}; "
            f"candidate={snapshot.get('candidate')}; "
            f"alignment={snapshot.get('alignment')}; "
            f"mode={snapshot.get('technical_mode')}; "
            f"indicative_spot={spot.get('price')}; "
            f"micro={micro.get('direction')}. "
            "All returned prices are research/indicative only; execution remains locked."
        )

        data = dict(snapshot)
        data["execution_eligible"] = False

        return ToolResult.success(
            summary=summary,
            data=data,
            sources=[
                {
                    "name": "XAUS live spot and intraday sampled series",
                    "url": "https://xaus.com/api/",
                },
                {
                    "name": "goldprice.dev indicative XAU/USD spot reference",
                    "url": "https://goldprice.dev/docs",
                },
                {
                    "name": "Yahoo Finance GC=F fallback reference",
                    "url": "https://finance.yahoo.com/quote/GC=F/",
                },
            ],
            observed_at=datetime.now(timezone.utc),
        )

    spec = ToolSpec(
        name="get_xau_intraday_research",
        title="XAU intraday research",
        description=(
            "Get the live PanWatch XAU/USD research snapshot. The primary intraday "
            "path uses an indicative live spot reference, a roughly two-minute spot "
            "micro-series, and sampled 5m/15m spot structure; Yahoo GC=F is fallback "
            "context only. Returns freshness gates, EMA/RSI/ATR structure, swings, "
            "alignment and a research candidate. Never use returned prices as broker "
            "execution quotes."
        ),
        risk=ToolRisk.READ,
        confirmation_required=False,
        exposure=ToolExposure.DEFERRED,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    registry.register(spec, get_xau_intraday_research)

    async def get_xau_decision_fusion(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        try:
            technical = await get_xau_snapshot(force=False)
            macro = await get_macro_context(force=False)
            fusion = build_decision_fusion(technical, macro)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(
                summary=f"XAU decision fusion unavailable: {type(exc).__name__}",
                error_code="xau_decision_fusion_unavailable",
            )

        summary = (
            "XAUUSD decision fusion: "
            f"state={fusion.get('state')}; "
            f"technical={fusion.get('technical_candidate')}; "
            f"macro_relation={fusion.get('macro_relation')}; "
            f"regime={fusion.get('regime')}; "
            f"confidence={fusion.get('cognitive_confidence')}; "
            f"meta={fusion.get('meta_decision')}; "
            f"event_risk={fusion.get('event_risk')}; "
            f"execution={fusion.get('execution_status')}. "
            "This is a research state, not an execution instruction."
        )
        return ToolResult.success(
            summary=summary,
            data={
                "fusion": fusion,
                "technical": technical,
                "macro": macro,
                "execution_eligible": False,
            },
            sources=[
                {
                    "name": "XAUS live spot and sampled intraday series",
                    "url": "https://xaus.com/api/",
                },
                {
                    "name": "goldprice.dev indicative XAU/USD spot reference",
                    "url": "https://goldprice.dev/docs",
                },
            ],
            observed_at=datetime.now(timezone.utc),
        )

    fusion_spec = ToolSpec(
        name="get_xau_decision_fusion",
        title="XAU full decision fusion",
        description=(
            "Combine the current XAU/USD technical research state with current macro "
            "research and explicit event/data gates. Returns support/conflict/neutral "
            "relationship and a deterministic research state without arbitrary scoring. "
            "Execution is always locked unless a separate tradable broker feed is added."
        ),
        risk=ToolRisk.READ,
        confirmation_required=False,
        exposure=ToolExposure.DEFERRED,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    registry.register(fusion_spec, get_xau_decision_fusion)

    async def run_gen1_trade_gold(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        """Run the frozen Ahmed ToolBox -> PanWatch -> Gen1 gold pipeline."""
        stage_errors: dict[str, str] = {}

        # Stage 1: force the macro refresh first. The macro service uses Ahmed
        # ToolBox reach_web_search as its primary discovery path and records the
        # actual search_source so fallback can never be hidden from the caller.
        try:
            macro = await asyncio.wait_for(
                get_macro_context(force=True),
                timeout=75.0,
            )
        except Exception as exc:  # noqa: BLE001 - fail visible, continue degraded
            stage_errors["ahmed_toolbox_macro"] = type(exc).__name__
            try:
                macro = await get_macro_context(force=False)
            except Exception as fallback_exc:  # noqa: BLE001
                stage_errors["macro_fallback"] = type(fallback_exc).__name__
                macro = {
                    "bias": 0,
                    "bias_label": "neutral",
                    "confidence": 0.0,
                    "event_risk": False,
                    "search_ok": False,
                    "search_source": None,
                    "synthesis_ok": False,
                    "summary": "Macro layer unavailable.",
                    "drivers": [],
                }

        # Stage 2: PanWatch builds the current technical/market state including
        # HTF context, SMC/liquidity and XAUT microstructure.
        try:
            technical = await get_xau_snapshot(force=False)
        except Exception as exc:  # noqa: BLE001
            stage_errors["panwatch"] = type(exc).__name__
            technical = {
                "instrument": "XAUUSD",
                "status": "unavailable",
                "candidate": "none",
                "blocked": True,
                "block_reasons": [f"panwatch_unavailable:{type(exc).__name__}"],
                "warnings": [],
                "frames": {},
                "execution_allowed": False,
                "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
            }

        # Stage 3: Gen1 consumes the PanWatch state plus the macro state. This
        # stage is deliberately last; it never performs a second independent
        # market-data search that could drift from the state above.
        try:
            fusion = build_decision_fusion(technical, macro)
        except Exception as exc:  # noqa: BLE001
            stage_errors["gen1"] = type(exc).__name__
            fusion = {
                "state": "unavailable",
                "technical_candidate": technical.get("candidate", "none"),
                "research_ready": False,
                "execution_allowed": False,
                "reasons": [f"gen1_fusion_unavailable:{type(exc).__name__}"],
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
        if not xaut or technical.get("xaut_order_flow_error"):
            missing_layers.append("xaut_order_flow")
        if not footprint.get("available"):
            missing_layers.append("xaut_footprint")
        if profile.get("status") != "ready":
            missing_layers.append("xaut_volume_profile")
        if not raw_book:
            missing_layers.append("xaut_raw_book")

        toolbox_stage = {
            "status": (
                "ready"
                if macro.get("search_ok") and search_source == "ahmed_toolbox"
                else "degraded"
            ),
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
            "xaut_ready": bool(xaut) and not technical.get("xaut_order_flow_error"),
            "footprint_ready": bool(footprint.get("available")),
            "volume_profile_ready": profile.get("status") == "ready",
            "raw_book_ready": bool(raw_book),
        }
        gen1_stage = {
            "status": fusion.get("state"),
            "candidate": fusion.get("technical_candidate"),
            "regime": fusion.get("regime"),
            "confidence": fusion.get("cognitive_confidence"),
            "meta_decision": fusion.get("meta_decision"),
            "research_ready": bool(fusion.get("research_ready")),
            "execution_allowed": False,
        }
        pipeline_status = "ready" if not missing_layers and not stage_errors else "degraded"

        summary = (
            "GEN1 TRADE GOLD pipeline: "
            f"status={pipeline_status}; "
            f"toolbox={toolbox_stage['status']}; "
            f"panwatch={panwatch_stage['status']}; "
            f"gen1={gen1_stage['status']}; "
            f"candidate={gen1_stage['candidate']}; "
            f"confidence={gen1_stage['confidence']}; "
            f"missing={','.join(missing_layers) if missing_layers else 'none'}. "
            "Research only; no live execution is enabled."
        )
        return ToolResult.success(
            summary=summary,
            data={
                "contract": "gen1-trade-gold-v1",
                "trigger": "Gen1 trade gold",
                "pipeline_order": ["ahmed_toolbox", "panwatch", "gen1"],
                "pipeline_status": pipeline_status,
                "stages": {
                    "ahmed_toolbox": toolbox_stage,
                    "panwatch": panwatch_stage,
                    "gen1": gen1_stage,
                },
                "missing_layers": missing_layers,
                "stage_errors": stage_errors,
                "macro": macro,
                "technical": technical,
                "fusion": fusion,
                "forward_range_map": xaut.get("forward_range_map"),
                "answer_contract": {
                    "required": [
                        "LONG_SHORT_WAIT",
                        "confidence",
                        "primary_scenario",
                        "alternative_scenario",
                        "invalidation",
                        "plus_minus_10_20_30",
                        "confirmations",
                        "missing_layers",
                    ],
                    "never_hide_missing_layer": True,
                    "never_treat_xaut_as_global_xauusd_order_flow": True,
                    "execution_allowed": False,
                },
            },
            sources=[
                {"name": "Ahmed ToolBox / macro research"},
                {"name": "PanWatch XAU market intelligence"},
                {"name": "Bitfinex XAUT/USD microstructure"},
            ],
            observed_at=datetime.now(timezone.utc),
        )

    gen1_spec = ToolSpec(
        name="run_gen1_trade_gold",
        title="GEN1 Trade Gold full pipeline",
        description=(
            "Run the frozen full gold research pipeline in strict order: Ahmed ToolBox "
            "macro/external evidence first, PanWatch market intelligence second, then "
            "Gen1 decision fusion. Includes HTF/EMA/SMC/liquidity plus XAUT CVD, "
            "footprint, raw order book, executed-volume profile and ±10/20/30 mapping. "
            "Missing layers are surfaced explicitly and live execution remains disabled."
        ),
        risk=ToolRisk.READ,
        confirmation_required=False,
        exposure=ToolExposure.DIRECT,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    registry.register(gen1_spec, run_gen1_trade_gold)

    async def get_xau_paper_league(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        try:
            paper_engine = XAUPaperTradingEngine()
            summary_data = await asyncio.to_thread(
                paper_engine.summary,
                trade_limit=50,
                signal_limit=50,
            )
            summary_data["weekly_history"] = await asyncio.to_thread(paper_engine.history, limit=12)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(
                summary=f"XAU paper league unavailable: {type(exc).__name__}",
                error_code="xau_paper_league_unavailable",
            )

        account = summary_data.get("account") or {}
        position = summary_data.get("position") or {}
        summary = (
            "XAU weekly paper league: "
            f"week={account.get('week_key')}; "
            f"equity={account.get('current_equity')}; "
            f"realized_pnl={account.get('realized_pnl')}; "
            f"trades={account.get('total_trades')}; "
            f"position={position.get('side') if position else 'flat'}; "
            f"history_weeks={len(summary_data.get('weekly_history') or [])}. "
            "Simulation only; no live execution capability."
        )
        return ToolResult.success(
            summary=summary,
            data=summary_data,
            sources=[],
            observed_at=datetime.now(timezone.utc),
        )

    paper_spec = ToolSpec(
        name="get_xau_paper_league",
        title="XAU weekly paper league",
        description=(
            "Read the current $10k weekly XAU/USD paper-trading account, open "
            "position, closed trades, MFE/MAE, R multiples, setup audit trail, "
            "spread guard and max-hold rules. This tool is simulation-only and "
            "can never route a live order."
        ),
        risk=ToolRisk.READ,
        confirmation_required=False,
        exposure=ToolExposure.DEFERRED,
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )
    registry.register(paper_spec, get_xau_paper_league)

    return [
        ToolDescriptor(
            tool_name=spec.name,
            title=spec.title,
            summary=(
                "Near-real-time XAU/USD research state from indicative spot micro "
                "data plus sampled 5m/15m structure; not valid for execution pricing."
            ),
            use_cases=[
                "gold intraday analysis",
                "XAUUSD technical context",
                "gold scalping research",
                "spot micro momentum",
            ],
            keywords=[
                "XAUUSD",
                "gold",
                "spot",
                "micro",
                "5m",
                "15m",
                "ATR",
                "RSI",
                "EMA",
                "GC=F",
            ],
            aliases=["gold research", "xau research", "xauusd intraday"],
            domain="market_research",
            capabilities=[
                "xau_intraday",
                "technical_analysis",
                "spot_micro",
                "sampled_spot_bars",
                "research_only",
            ],
            data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
            estimated_latency_ms=3_500,
            output_summary="Research-only deterministic XAU intraday state.",
            risk=ToolRisk.READ,
            confirmation_required=False,
            implementation_version="xau-research-0.2",
        ),
        ToolDescriptor(
            tool_name=fusion_spec.name,
            title=fusion_spec.title,
            summary=(
                "Full XAU cognition across market perception, regime detection, "
                "competing hypotheses, macro context, adversarial review, calibrated "
                "confidence, execution timing and explicit data/event gates; research-only."
            ),
            use_cases=[
                "full gold analysis",
                "XAUUSD macro technical fusion",
                "gold trade research",
                "event risk check",
            ],
            keywords=[
                "XAUUSD",
                "gold",
                "macro",
                "event risk",
                "fusion",
                "decision",
                "technical",
            ],
            aliases=["xau fusion", "gold full analysis", "xau decision"],
            domain="market_research",
            capabilities=[
                "xau_decision_fusion",
                "macro_context",
                "event_gate",
                "technical_analysis",
                "research_only",
            ],
            data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
            estimated_latency_ms=12_000,
            output_summary="Research-only XAU technical + macro decision fusion.",
            risk=ToolRisk.READ,
            confirmation_required=False,
            implementation_version="xau-fusion-1.0",
        )
,
        ToolDescriptor(
            tool_name=gen1_spec.name,
            title=gen1_spec.title,
            summary=(
                "Deterministic Ahmed ToolBox → PanWatch → Gen1 gold pipeline with "
                "fail-visible layer health and XAUT microstructure."
            ),
            use_cases=[
                "Gen1 trade gold",
                "full gold trading research",
                "gold scalping full pipeline",
            ],
            keywords=[
                "Gen1 trade gold",
                "GEN1",
                "XAUUSD",
                "gold",
                "Ahmed Toolbox",
                "PanWatch",
                "order flow",
                "footprint",
                "volume profile",
            ],
            aliases=["Gen1 trade gold", "gen1 trade gold"],
            domain="market_research",
            capabilities=[
                "gen1_trade_gold",
                "full_gold_pipeline",
                "macro_context",
                "xau_market_intelligence",
                "xaut_order_flow",
                "research_only",
            ],
            data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
            estimated_latency_ms=45_000,
            output_summary="Full GEN1 gold research pipeline with explicit layer health.",
            risk=ToolRisk.READ,
            confirmation_required=False,
            implementation_version="gen1-trade-gold-1.0",
        ),
        ToolDescriptor(
            tool_name=paper_spec.name,
            title=paper_spec.title,
            summary=(
                "Read-only weekly $10k XAU paper league performance, open risk, "
                "MFE/MAE and setup audit trail."
            ),
            use_cases=[
                "gold paper trading performance",
                "weekly XAU demo league",
                "review XAU trades",
                "MFE MAE analysis",
            ],
            keywords=[
                "XAUUSD",
                "gold",
                "paper trading",
                "demo",
                "weekly",
                "MFE",
                "MAE",
                "R multiple",
            ],
            aliases=["xau paper", "gold demo", "paper league"],
            domain="portfolio_research",
            capabilities=[
                "xau_paper_trading",
                "performance_review",
                "trade_audit",
                "research_only",
            ],
            data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
            estimated_latency_ms=200,
            output_summary="Read-only XAU paper league account and trade history.",
            risk=ToolRisk.READ,
            confirmation_required=False,
            implementation_version="xau-paper-0.1",
        )
    ]
