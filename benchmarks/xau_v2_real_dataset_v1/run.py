"""Real-network Biquote/MT5 dataset smoke for XAU Strategy v2.

This job validates acquisition/provenance and deep HTF coverage. Its deliberately
short M1 window MUST keep edge_claim_ready=false.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.modules.strategy.xau_v2_dataset import (
    BiquoteXAUV2DatasetBuilder,
    XAUV2DatasetPlan,
)


def main() -> None:
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    plan = XAUV2DatasetPlan.real_smoke(end)
    result = BiquoteXAUV2DatasetBuilder().build(plan)
    audit = result.audit
    payload = audit.to_dict()

    deep_ema = all(
        payload["frames"][name]["ema1000_supported"]
        for name in ("1h", "4h", "1d")
    )
    source_identity = all(
        frame["symbol_set"] == ["XAUUSD"]
        and all(
            str(source).lower().startswith("biquote.io:mt5-ohlc")
            for source in frame["source_set"]
        )
        for frame in payload["frames"].values()
    )

    payload["benchmark"] = "xau-v2-real-dataset-smoke-v1"
    payload["smoke_contract"] = {
        "deep_ema1000_supported": deep_ema,
        "source_identity_consistent": source_identity,
        "short_intraday_window_must_not_allow_edge_claim": (
            audit.edge_claim_ready is False
        ),
        "real_network": True,
        "synthetic_data": False,
    }
    payload["passed"] = bool(
        audit.valid
        and audit.wiring_ready
        and source_identity
        and audit.edge_claim_ready is False
        and audit.deep_history_source_required is True
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
