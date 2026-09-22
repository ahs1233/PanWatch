from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from src.modules.strategy.xau_v2_dataset import (
    BiquoteXAUV2DatasetBuilder,
    XAUV2DatasetPlan,
    XAUV2DatasetReadiness,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


UTC = timezone.utc
END = datetime(2026, 9, 23, 18, 0, tzinfo=UTC)

_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.H1: timedelta(hours=1),
    XAUTimeframe.H4: timedelta(hours=4),
    XAUTimeframe.D1: timedelta(days=1),
}


class FakeBiquoteProvider:
    def __init__(
        self,
        *,
        bad_symbol: XAUTimeframe | None = None,
        bad_source: XAUTimeframe | None = None,
        duplicate: XAUTimeframe | None = None,
    ) -> None:
        self.bad_symbol = bad_symbol
        self.bad_source = bad_source
        self.duplicate = duplicate
        self.calls: list[dict] = []

    def bars_range(
        self,
        timeframe: XAUTimeframe,
        *,
        start: datetime,
        end: datetime,
        timeout_seconds: float = 12.0,
        max_bars_per_request: int = 900,
        max_chunks: int = 64,
    ) -> list[XAUBar]:
        self.calls.append(
            {
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "timeout_seconds": timeout_seconds,
                "max_bars_per_request": max_bars_per_request,
                "max_chunks": max_chunks,
            }
        )
        delta = _DURATION[timeframe]
        rows: list[XAUBar] = []
        timestamp = start
        index = 0
        previous = 4300.0
        while timestamp < end:
            close = 4300.0 + index * 0.02
            symbol = "GC=F" if timeframe is self.bad_symbol else "XAUUSD"
            source = (
                "unexpected-provider"
                if timeframe is self.bad_source
                else "biquote.io:MT5-ohlc"
            )
            rows.append(
                XAUBar(
                    timestamp=timestamp,
                    timeframe=timeframe,
                    open=previous,
                    high=max(previous, close) + 0.2,
                    low=min(previous, close) - 0.2,
                    close=close,
                    volume=1000 + index % 50,
                    source=source,
                    symbol=symbol,
                    execution_eligible=False,
                )
            )
            previous = close
            timestamp += delta
            index += 1

        if self.duplicate is timeframe and rows:
            rows.append(rows[-1])
        return rows


def _plan(
    *,
    min_intraday_research_days: float = 365.0,
) -> XAUV2DatasetPlan:
    return XAUV2DatasetPlan(
        end=END,
        lookback_days={
            XAUTimeframe.M1: 2.0,
            XAUTimeframe.M5: 4.0,
            XAUTimeframe.M15: 10.0,
            XAUTimeframe.H1: 60.0,
            XAUTimeframe.H4: 200.0,
            XAUTimeframe.D1: 1200.0,
        },
        minimum_bars={
            XAUTimeframe.M1: 1000,
            XAUTimeframe.M5: 500,
            XAUTimeframe.M15: 500,
            XAUTimeframe.H1: 1000,
            XAUTimeframe.H4: 1000,
            XAUTimeframe.D1: 1000,
        },
        minimum_intraday_research_days=min_intraday_research_days,
        max_chunks_per_timeframe=16,
        dataset_id="fake-biquote-ci",
        profile="unit_test",
    )


def test_builder_fetches_all_required_frames_with_asymmetric_windows():
    provider = FakeBiquoteProvider()
    result = BiquoteXAUV2DatasetBuilder(provider).build(_plan())

    assert len(provider.calls) == 6
    by_frame = {call["timeframe"]: call for call in provider.calls}
    assert set(by_frame) == {
        XAUTimeframe.M1,
        XAUTimeframe.M5,
        XAUTimeframe.M15,
        XAUTimeframe.H1,
        XAUTimeframe.H4,
        XAUTimeframe.D1,
    }

    assert END - by_frame[XAUTimeframe.M1]["start"] == timedelta(days=2)
    assert END - by_frame[XAUTimeframe.H1]["start"] == timedelta(days=60)
    assert END - by_frame[XAUTimeframe.D1]["start"] == timedelta(days=1200)

    assert result.audit.valid is True
    assert result.audit.wiring_ready is True
    assert result.audit.readiness is XAUV2DatasetReadiness.WIRING_VALID
    assert result.audit.edge_claim_ready is False
    assert any(
        "intraday_history_too_short" in warning
        for warning in result.audit.warnings
    )


def test_deep_htf_counts_explicitly_support_ema1000():
    result = BiquoteXAUV2DatasetBuilder(FakeBiquoteProvider()).build(_plan())
    frames = result.audit.frames

    assert frames["1h"].ema1000_supported is True
    assert frames["4h"].ema1000_supported is True
    assert frames["1d"].ema1000_supported is True


def test_same_data_can_be_research_candidate_only_when_intraday_horizon_gate_is_met():
    result = BiquoteXAUV2DatasetBuilder(FakeBiquoteProvider()).build(
        _plan(min_intraday_research_days=1.0)
    )
    assert result.audit.valid is True
    assert result.audit.wiring_ready is True
    assert result.audit.edge_claim_ready is True
    assert result.audit.readiness is XAUV2DatasetReadiness.RESEARCH_CANDIDATE


def test_builder_rejects_cross_instrument_spot_frame_instead_of_silent_proxy_fallback():
    provider = FakeBiquoteProvider(bad_symbol=XAUTimeframe.M15)
    with pytest.raises(ValueError, match="non-XAUUSD"):
        BiquoteXAUV2DatasetBuilder(provider).build(_plan())


def test_builder_rejects_unexpected_provider_family():
    provider = FakeBiquoteProvider(bad_source=XAUTimeframe.H1)
    with pytest.raises(ValueError, match="provider_family_mismatch_1h"):
        BiquoteXAUV2DatasetBuilder(provider).build(_plan())


def test_builder_rejects_duplicate_timestamps():
    provider = FakeBiquoteProvider(duplicate=XAUTimeframe.M5)
    with pytest.raises(ValueError, match="duplicate_5m_timestamps"):
        BiquoteXAUV2DatasetBuilder(provider).build(_plan())


def test_dataset_fingerprint_is_stable_for_identical_provider_results():
    a = BiquoteXAUV2DatasetBuilder(FakeBiquoteProvider()).build(_plan())
    b = BiquoteXAUV2DatasetBuilder(FakeBiquoteProvider()).build(_plan())
    assert a.dataset.fingerprint == b.dataset.fingerprint
    assert a.audit.dataset_fingerprint == b.audit.dataset_fingerprint


def test_dataset_fingerprint_changes_with_underlying_bar_data():
    provider = FakeBiquoteProvider()
    original = BiquoteXAUV2DatasetBuilder(provider).build(_plan())

    class ChangedProvider(FakeBiquoteProvider):
        def bars_range(self, timeframe, **kwargs):
            rows = super().bars_range(timeframe, **kwargs)
            if timeframe is XAUTimeframe.M1 and rows:
                row = rows[-1]
                close = row.close + 1.0
                rows[-1] = replace(
                    row,
                    high=max(row.high, close + 0.1),
                    close=close,
                )
            return rows

    changed = BiquoteXAUV2DatasetBuilder(ChangedProvider()).build(_plan())
    assert original.dataset.fingerprint != changed.dataset.fingerprint


def test_audit_manifest_keeps_provider_and_instrument_identity_explicit():
    result = BiquoteXAUV2DatasetBuilder(FakeBiquoteProvider()).build(_plan())
    payload = result.audit.to_dict()

    assert payload["instrument"] == "XAUUSD"
    assert payload["provider_family"] == "biquote.io:MT5-ohlc"
    assert payload["edge_claim_ready"] is False
    for frame in payload["frames"].values():
        assert frame["symbol_set"] == ["XAUUSD"]
        assert frame["source_set"] == ["biquote.io:MT5-ohlc"]
