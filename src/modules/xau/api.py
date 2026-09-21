"""HTTP API for the XAU/USD research terminal."""

from fastapi import APIRouter, HTTPException, Query

from .service import get_macro_context, get_xau_snapshot

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
            "execution_status": "LOCKED_NO_SPOT_FEED",
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
    return {"technical": technical, "macro": macro_context}
