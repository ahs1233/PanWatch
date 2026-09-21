"""Research-only XAUUSD tools for the PanWatch assistant."""

from __future__ import annotations

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

    async def get_xau_paper_league(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        try:
            paper_engine = XAUPaperTradingEngine()
            summary_data = paper_engine.summary(
                trade_limit=50,
                signal_limit=50,
            )
            summary_data["weekly_history"] = paper_engine.history(limit=12)
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
            "position, closed trades, MFE/MAE, R multiples and setup audit trail. "
            "This tool is simulation-only and can never route a live order."
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
                "Full XAU research fusion across live technical structure, macro "
                "context and explicit data/event gates; research-only."
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
            implementation_version="xau-fusion-0.1",
        )
,
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
