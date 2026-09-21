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

from src.modules.xau.service import get_xau_snapshot


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
        )
    ]
