"""Extract LEAN output and compare it with the PanWatch Native result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MARKER = "PANWATCH_PARITY_RESULT="
NUMERIC_FIELDS = ("ema_fast", "ema_slow", "rsi14", "atr14", "close")
TOLERANCE = 1e-6


def extract_lean_result(log_path: Path) -> dict[str, Any]:
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if MARKER not in line:
            continue
        payload = line.split(MARKER, 1)[1].strip()
        return json.loads(payload)
    raise RuntimeError("LEAN log did not contain PANWATCH_PARITY_RESULT")


def compare(native: dict[str, Any], lean: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []

    def exact(field: str) -> None:
        if native.get(field) != lean.get(field):
            mismatches.append(
                {"field": field, "native": native.get(field), "lean": lean.get(field)}
            )

    for field in (
        "dataset_version",
        "dataset_fingerprint",
        "strategy_fingerprint",
        "status",
        "candidate",
        "blocked",
        "block_reasons",
    ):
        exact(field)

    native_frames = native.get("frame_states") or {}
    lean_frames = lean.get("frame_states") or {}
    if set(native_frames) != set(lean_frames):
        mismatches.append(
            {
                "field": "frame_states.keys",
                "native": sorted(native_frames),
                "lean": sorted(lean_frames),
            }
        )

    deltas: dict[str, dict[str, float]] = {}
    for timeframe in sorted(set(native_frames) & set(lean_frames)):
        left = native_frames[timeframe]
        right = lean_frames[timeframe]
        for field in ("direction", "breakout"):
            if left.get(field) != right.get(field):
                mismatches.append(
                    {
                        "field": f"{timeframe}.{field}",
                        "native": left.get(field),
                        "lean": right.get(field),
                    }
                )
        deltas[timeframe] = {}
        for field in NUMERIC_FIELDS:
            delta = abs(float(left[field]) - float(right[field]))
            deltas[timeframe][field] = delta
            if delta > TOLERANCE:
                mismatches.append(
                    {
                        "field": f"{timeframe}.{field}",
                        "native": left[field],
                        "lean": right[field],
                        "delta": delta,
                        "tolerance": TOLERANCE,
                    }
                )

    return {
        "benchmark": "xau-engine-parity-v1",
        "passed": not mismatches,
        "tolerance": TOLERANCE,
        "candidate": {
            "native": native.get("candidate"),
            "lean": lean.get("candidate"),
        },
        "numeric_deltas": deltas,
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--lean-log", type=Path, required=True)
    parser.add_argument("--lean-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    native = json.loads(args.native.read_text(encoding="utf-8"))
    lean = extract_lean_result(args.lean_log)
    args.lean_output.write_text(json.dumps(lean, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = compare(native, lean)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
