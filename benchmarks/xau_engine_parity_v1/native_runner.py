"""Run the canonical XAU strategy through PanWatch's native engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.strategy.xau_strategy_exporter import (
    export_backend_package,
    export_xau_intraday_strategy,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_bars(dataset: dict[str, Any]) -> dict[XAUTimeframe, list[XAUBar]]:
    out: dict[XAUTimeframe, list[XAUBar]] = {}
    for raw_timeframe, rows in dataset["bars"].items():
        timeframe = XAUTimeframe(raw_timeframe)
        out[timeframe] = [
            XAUBar(
                timestamp=_parse_time(row["timestamp"]),
                timeframe=timeframe,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                source="panwatch:frozen-parity-v1",
                execution_eligible=False,
            )
            for row in rows
        ]
    return out


def _frame_payload(state) -> dict[str, Any]:
    return {
        "direction": state.direction,
        "breakout": state.breakout,
        "ema_fast": round(float(state.ema_fast), 10),
        "ema_slow": round(float(state.ema_slow), 10),
        "rsi14": round(float(state.rsi14), 10),
        "atr14": round(float(state.atr14), 10),
        "close": round(float(state.close), 10),
    }


def run(dataset_path: Path, output_path: Path, lean_package_path: Path) -> dict[str, Any]:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    bars = _load_bars(dataset)
    engine = XAUIntradayEngine()
    spec = export_xau_intraday_strategy(engine)
    latest = max(rows[-1].timestamp for rows in bars.values())

    assessment = engine.analyze(
        bars,
        event_risk=False,
        macro_bias=0,
        now=latest,
    )
    result = {
        "backend": "native",
        "dataset_version": dataset["dataset_version"],
        "dataset_fingerprint": _canonical_hash(dataset),
        "strategy_fingerprint": spec.fingerprint,
        "status": assessment.status,
        "candidate": assessment.candidate,
        "blocked": assessment.blocked,
        "block_reasons": list(assessment.block_reasons),
        "frame_states": {
            timeframe: _frame_payload(state)
            for timeframe, state in sorted(assessment.frame_states.items())
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lean_package = export_backend_package("lean", spec=spec)
    lean_package["dataset_version"] = dataset["dataset_version"]
    lean_package["dataset_fingerprint"] = result["dataset_fingerprint"]
    lean_package_path.parent.mkdir(parents=True, exist_ok=True)
    lean_package_path.write_text(
        json.dumps(lean_package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lean-package", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.output, args.lean_package)


if __name__ == "__main__":
    main()
