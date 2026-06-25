"""
Structured-output schema for LLM churn-signal extraction.

A single ChurnSignal captures the 6 risk dimensions extracted from one review.
ChurnSignalBatch wraps a list of them and enforces that the LLM returns one
object per input review, in the same order.

These models are consumed by `features.llm_extractor` (via the `instructor`
library, which uses them as `response_model`) and by the pytest suite.
"""
from typing import List

from pydantic import BaseModel, Field


class ChurnSignal(BaseModel):
    """Structured churn-risk signals for a single review."""

    sentiment_score: float = Field(
        ge=-1.0, le=1.0,
        description="Overall sentiment polarity from -1 (very negative) to +1 (very positive).",
    )
    frustration_level: float = Field(
        ge=0.0, le=1.0,
        description="Intensity of user frustration from 0 (calm) to 1 (very frustrated).",
    )
    technical_issue: bool = Field(
        description="True if the review mentions bugs, crashes, lag, errors, or performance problems.",
    )
    value_complaint: bool = Field(
        description="True if the review mentions pricing, cost, pay-to-win, or not worth money.",
    )
    engagement_dropped: bool = Field(
        description="True if the user mentions stopping play, giving up, uninstalling, or losing interest.",
    )
    positive_signal: bool = Field(
        description="True if the user recommends the product, loves it, or intends to keep playing.",
    )


class ChurnSignalBatch(BaseModel):
    """A list of ChurnSignal objects, one per input review, in the same order."""

    signals: List[ChurnSignal] = Field(
        description="One ChurnSignal per input review, preserving input order.",
    )


NEUTRAL_SIGNAL = ChurnSignal(
    sentiment_score=0.0,
    frustration_level=0.0,
    technical_issue=False,
    value_complaint=False,
    engagement_dropped=False,
    positive_signal=False,
)
