"""Tests for the Pydantic-typed LLM extractor (schema + alignment)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from features.llm_extractor import _align_batch
from features.schema import NEUTRAL_SIGNAL, ChurnSignal


def test_schema_rejects_sentiment_out_of_range():
    with pytest.raises(ValidationError):
        ChurnSignal(
            sentiment_score=5.0,
            frustration_level=0.0,
            technical_issue=False,
            value_complaint=False,
            engagement_dropped=False,
            positive_signal=False,
        )


def test_schema_rejects_string_for_float():
    with pytest.raises(ValidationError):
        ChurnSignal(
            sentiment_score="not a number",  # type: ignore[arg-type]
            frustration_level=0.0,
            technical_issue=False,
            value_complaint=False,
            engagement_dropped=False,
            positive_signal=False,
        )


def test_align_batch_truncates_extra_items():
    sigs = [NEUTRAL_SIGNAL] * 12
    aligned = _align_batch(sigs, expected_n=10)
    assert len(aligned) == 10


def test_align_batch_pads_missing_items_and_preserves_order():
    positive = ChurnSignal(
        sentiment_score=0.9,
        frustration_level=0.0,
        technical_issue=False,
        value_complaint=False,
        engagement_dropped=False,
        positive_signal=True,
    )
    sigs = [positive] * 7
    aligned = _align_batch(sigs, expected_n=10)
    assert len(aligned) == 10
    assert all(s.positive_signal for s in aligned[:7])
    assert all(not s.positive_signal for s in aligned[7:])
    assert all(s.sentiment_score == 0.0 for s in aligned[7:])


def test_align_batch_passthrough_when_exact_match():
    sigs = [NEUTRAL_SIGNAL] * 5
    aligned = _align_batch(sigs, expected_n=5)
    assert aligned is sigs or aligned == sigs
