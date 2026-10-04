from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Side = Literal["up", "down"]


class GtgEvent(BaseModel):
    """Domain event only; no execution authority."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    event_time: datetime
    side: Side
    price: float = Field(gt=0)
    atr: float = Field(gt=0)


class EventPrediction(BaseModel):
    """Prediction captured at event time before the outcome exists."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    recorded_at: datetime
    candidate_version: str
    mfe_atr: float = Field(ge=0)
    mae_atr: float = Field(ge=0)
    baseline_mfe_atr: float = Field(ge=0)
    baseline_mae_atr: float = Field(ge=0)


class ForwardOutcome(BaseModel):
    """Observed path after the fixed forward horizon completes."""

    model_config = ConfigDict(frozen=True)

    event_id: str
    horizon_completed_at: datetime
    actual_mfe_atr: float = Field(ge=0)
    actual_mae_atr: float = Field(ge=0)
