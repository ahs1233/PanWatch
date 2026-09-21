"""HTTP API for the XAU/USD research terminal."""

from fastapi import APIRouter, HTTPException, Query

from .paper import XAUPaperTradingEngine
from .service import build_decision_fusion, get_macro_context, get_xau_snapshot

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
async def paper_summary(
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
async def paper_weeks(
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
