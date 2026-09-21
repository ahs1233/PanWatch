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

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.platform.marketdata.xau_models import XAUTimeframe
from src.platform.marketdata.xau_research_provider import YahooGoldResearchProvider


def register_xau_research_tools(registry: ToolRegistry) -> list[ToolDescriptor]:
    """Register research-only XAU tooling without implying spot execution data."""

    async def get_xau_intraday_research(
        _request: RunRequest,
        _arguments: dict,
    ) -> ToolResult:
        provider = YahooGoldResearchProvider()

        async def _bars(timeframe: XAUTimeframe):
            return await asyncio.to_thread(provider.bars, timeframe)

        try:
            m1, m5, m15 = await asyncio.gather(
                _bars(XAUTimeframe.M1),
                _bars(XAUTimeframe.M5),
                _bars(XAUTimeframe.M15),
            )
        except Exception as exc:  # noqa: BLE001 - provider failure is a tool result
            return ToolResult.failure(
                summary=f"XAU research proxy data unavailable: {exc}",
                error_code="xau_research_unavailable",
            )

        assessment = XAUIntradayEngine(require_execution_data=False).analyze(
            {
                XAUTimeframe.M1: m1,
                XAUTimeframe.M5: m5,
                XAUTimeframe.M15: m15,
            },
            now=datetime.now(timezone.utc),
        )

        frames = {
            name: {
                "timeframe": state.timeframe.value,
                "close": state.close,
                "ema_fast": state.ema_fast,
                "ema_slow": state.ema_slow,
                "rsi14": state.rsi14,
                "atr14": state.atr14,
                "atr_pct": state.atr_pct,
                "breakout": state.breakout,
                "direction": state.direction,
                "recent_swing_high": state.recent_swing_high,
                "recent_swing_low": state.recent_swing_low,
                "observed_at": state.observed_at.isoformat(),
            }
            for name, state in assessment.frame_states.items()
        }

        summary = (
            "XAUUSD research proxy (GC=F) intraday state: "
            f"{assessment.status}; candidate={assessment.candidate}. "
            "This is research context only, not a spot XAUUSD execution quote."
        )
        return ToolResult.success(
            summary=summary,
            data={
                "instrument": "XAUUSD",
                "research_proxy": "GC=F",
                "execution_eligible": False,
                "status": assessment.status,
                "candidate": assessment.candidate,
                "blocked": assessment.blocked,
                "block_reasons": list(assessment.block_reasons),
                "warnings": list(assessment.warnings),
                "atr_reference": assessment.atr_reference,
                "swing_high_reference": assessment.swing_high_reference,
                "swing_low_reference": assessment.swing_low_reference,
                "frames": frames,
            },
            sources=[
                {
                    "name": "Yahoo Finance GC=F research proxy",
                    "url": "https://finance.yahoo.com/quote/GC=F/",
                }
            ],
            observed_at=datetime.now(timezone.utc),
        )

    spec = ToolSpec(
        name="get_xau_intraday_research",
        title="XAU intraday research",
        description=(
            "Get a deterministic 1m/5m/15m gold research snapshot using Yahoo "
            "GC=F as a non-execution proxy. Returns EMA, RSI, ATR, breakout, "
            "swing levels, freshness gates and a candidate directional setup. "
            "Never treat the returned proxy price as a spot XAUUSD execution price."
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
                "Research-only XAUUSD 1m/5m/15m technical state from the GC=F "
                "Yahoo proxy; not valid for execution pricing."
            ),
            use_cases=[
                "gold intraday analysis",
                "XAUUSD technical context",
                "gold scalping research",
            ],
            keywords=[
                "XAUUSD",
                "gold",
                "GC=F",
                "1m",
                "5m",
                "15m",
                "ATR",
                "RSI",
                "EMA",
            ],
            aliases=["gold research", "xau research", "xauusd intraday"],
            domain="market_research",
            capabilities=[
                "xau_intraday",
                "technical_analysis",
                "research_proxy",
            ],
            data_freshness=ToolDataFreshness.NEAR_REAL_TIME,
            estimated_latency_ms=4_000,
            output_summary="Research-only deterministic XAU intraday state.",
            risk=ToolRisk.READ,
            confirmation_required=False,
            implementation_version="xau-research-0.1",
        )
    ]
