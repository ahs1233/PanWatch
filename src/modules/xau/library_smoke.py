"""Build-time functional smoke for the XAU intelligence stack."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

from src.modules.xau.library_intelligence import library_consensus, vectorbt_validation
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _bars(count: int = 360) -> list[XAUBar]:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows: list[XAUBar] = []
    previous = 4300.0
    for i in range(count):
        drift = i * 0.035
        wave = 9.0 * math.sin(i / 11.0) + 3.0 * math.sin(i / 3.7)
        close = 4300.0 + drift + wave
        open_ = previous
        high = max(open_, close) + 1.2 + 0.4 * abs(math.sin(i / 2.0))
        low = min(open_, close) - 1.1 - 0.3 * abs(math.cos(i / 2.3))
        rows.append(
            XAUBar(
                timestamp=start + timedelta(hours=i),
                timeframe=XAUTimeframe.H1,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=900.0 + (i % 24) * 37.0 + 120.0 * abs(math.sin(i / 5.0)),
                source="smoke:synthetic",
                symbol="XAUUSD",
                execution_eligible=False,
            )
        )
        previous = close
    return rows


def main() -> None:
    rows = _bars()
    consensus = library_consensus(rows)
    backtest = vectorbt_validation(rows)
    required = {
        "pyvsmc",
        "smartmoneyconcepts",
        "smc_mcp",
        "pandas_ta_classic",
        "marketprofile",
        "structure_scope_reference",
    }
    statuses = consensus.get("status") or {}
    failed = {
        name: statuses.get(name)
        for name in required
        if statuses.get(name) != "ok"
    }
    if backtest.get("status") != "ok":
        failed["vectorbt"] = backtest.get("status")
    if failed:
        raise RuntimeError(f"XAU library functional smoke failed: {failed}")
    print(
        "XAU_LIBRARY_FUNCTIONAL_OK="
        + json.dumps(
            {
                "status": statuses,
                "consensus_direction": consensus.get("direction"),
                "agreement": consensus.get("agreement"),
                "vectorbt": backtest.get("status"),
                "strategies": len(backtest.get("strategies") or []),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
