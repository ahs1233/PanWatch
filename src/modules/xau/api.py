"""HTTP API for the XAU/USD research terminal."""

from fastapi import APIRouter, HTTPException, Query

from .gen1_pipeline import run_gen1_trade_gold_pipeline
from .paper import XAUPaperTradingEngine
from .validation import load_gen1_live_validation_summary, load_gen1_validation_summary
from .service import build_decision_fusion, get_chart_series, get_library_validation, get_macro_context, get_xau_snapshot

router = APIRouter()


@router.get("/snapshot")
async def snapshot(force: bool = Query(default=False)):
    try:
        return await get_xau_snapshot(force=force)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"XAU research data unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/macro")
async def macro(force: bool = Query(default=False)):
    return await get_macro_context(force=force)


@router.get("/chart")
async def chart(
    timeframe: str = Query(default="5m", pattern="^(1m|5m|15m|30m|1h|4h|1d|1w|1mo)$"),
    limit: int = Query(default=160, ge=30, le=240),
    force: bool = Query(default=False),
):
    try:
        return await get_chart_series(timeframe=timeframe, limit=limit, force=force)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"XAU chart data unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/library-validation")
async def library_validation(force: bool = Query(default=False)):
    try:
        return await get_library_validation(force=force)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"XAU library validation unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/gen1-gold")
async def gen1_gold():
    """PanWatch UI surface for the same core used by 'Gen1 trade gold'."""
    try:
        return await run_gen1_trade_gold_pipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"GEN1 GOLD pipeline unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/gen1-gold/validation")
def gen1_gold_validation(limit: int = Query(default=2000, ge=10, le=10000)):
    """Historical validation of persisted GEN1 replay episodes."""
    try:
        return load_gen1_validation_summary(limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"GEN1 GOLD validation unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/gen1-gold/live-validation")
def gen1_gold_live_validation(limit: int = Query(default=2000, ge=10, le=10000)):
    """Forward validation for actual live GEN1 invocations, including XAUT."""
    try:
        return load_gen1_live_validation_summary(limit=limit)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"GEN1 GOLD live validation unavailable: {type(exc).__name__}",
        ) from exc


@router.get("/terminal")
async def terminal(force: bool = Query(default=False)):
    try:
        technical = await get_xau_snapshot(force=force)
    except Exception as exc:
        technical = {
            "instrument": "XAUUSD",
            "research_only": True,
            "execution_feed_connected": False,
            "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
            "status": "unavailable",
            "candidate": "none",
            "blocked": True,
            "block_reasons": [
                f"research_data_unavailable:{type(exc).__name__}"
            ],
            "warnings": [],
            "frames": {},
        }
    macro_context = await get_macro_context(force=force)
    fusion = build_decision_fusion(technical, macro_context)
    return {"technical": technical, "macro": macro_context, "fusion": fusion}



@router.get("/paper/summary")
def paper_summary(
    trade_limit: int = Query(default=30, ge=1, le=200),
    signal_limit: int = Query(default=30, ge=1, le=200),
):
    return XAUPaperTradingEngine().summary(
        trade_limit=trade_limit,
        signal_limit=signal_limit,
    )


@router.post("/paper/scan")
async def paper_scan():
    """Run one paper-only scan. Never routes a live order."""
    return await XAUPaperTradingEngine().scan()



@router.get("/paper/weeks")
def paper_weeks(
    limit: int = Query(default=12, ge=1, le=52),
):
    return {
        "weeks": XAUPaperTradingEngine().history(limit=limit),
        "execution_allowed": False,
    }



@router.get("/paper/eligibility")
async def paper_eligibility():
    """Explain the current paper entry state using the exact engine gates."""
    return await XAUPaperTradingEngine().eligibility()
