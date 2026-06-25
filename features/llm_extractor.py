"""
Extract structured churn-risk signals from review text via Groq + instructor.

The previous implementation hand-rolled JSON parsing, neutral-padding on count
mismatch, and a regex-based retry-delay extractor. This version uses:

* Pydantic schemas (`ChurnSignal`, `ChurnSignalBatch` in `features.schema`) for
  type-safe LLM outputs validated at the boundary;
* `instructor.from_groq(...)` to enforce those schemas on every response;
* `tenacity` for exponential-backoff retries on rate-limit / API errors.

Alignment is now explicit: `_align_batch` truncates or pads (loudly) so a
silent count mismatch in any one batch can no longer drift the entire dataset.

Public API preserved:
    extract_llm_features(df, save=True) -> pd.DataFrame  # bulk pipeline
    extract_single(text) -> dict                         # used by Streamlit
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import instructor
import pandas as pd
from groq import APIConnectionError, APIError, APIStatusError, Groq, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (  # noqa: E402
    FEATURES_PATH,
    GROQ_API_KEY,
    LLM_BATCH_SIZE,
    LLM_MAX_TOKENS,
    LLM_MODEL,
)
from features.schema import NEUTRAL_SIGNAL, ChurnSignal, ChurnSignalBatch  # noqa: E402

RATE_LIMIT_SLEEP = 2.0   # seconds between batches; tuned to 30K TPM Groq free tier
MAX_RETRIES      = 6
TEXT_TRUNCATE    = 600   # chars per review fed to the LLM

SYSTEM_PROMPT = (
    "You are a churn-risk analyst. For each customer review provided, "
    "extract one ChurnSignal object. Return exactly one object per input "
    "review, in the same order. No commentary, no markdown, just the "
    "structured signals."
)


# ── Per-batch call (retried on transient errors) ─────────────────────────────


# Only retry on transient Groq errors. ValidationError, KeyboardInterrupt, and
# our own programming bugs should bubble up immediately rather than be silently
# retried six times.
@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((
        RateLimitError, APIConnectionError, APIStatusError, APIError,
    )),
    reraise=True,
)
def _extract_batch_typed(client, texts: list[str]) -> ChurnSignalBatch:
    numbered = "\n\n".join(f"[{i + 1}] {t[:TEXT_TRUNCATE]}" for i, t in enumerate(texts))
    user_msg = f"Extract churn signals from these {len(texts)} reviews:\n\n{numbered}"
    return client.chat.completions.create(
        model=LLM_MODEL,
        response_model=ChurnSignalBatch,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )


def _align_batch(signals: list[ChurnSignal], expected_n: int) -> list[ChurnSignal]:
    """Truncate or pad with neutral signals so the returned list has length `expected_n`.

    Logs a warning when adjustment is needed, a silent fix would re-introduce
    the alignment-drift bug we just removed.
    """
    if len(signals) > expected_n:
        tqdm.write(
            f"  Warning: LLM returned {len(signals)} signals for batch of "
            f"{expected_n}; truncating to {expected_n}."
        )
        return signals[:expected_n]
    if len(signals) < expected_n:
        tqdm.write(
            f"  Warning: LLM returned {len(signals)} signals for batch of "
            f"{expected_n}; padding {expected_n - len(signals)} with neutral."
        )
        return signals + [NEUTRAL_SIGNAL] * (expected_n - len(signals))
    return signals


# ── Public API ───────────────────────────────────────────────────────────────


def _make_client() -> instructor.Instructor:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in .env")
    return instructor.from_groq(Groq(api_key=GROQ_API_KEY), mode=instructor.Mode.JSON)


def extract_llm_features(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """Run the LLM extractor over every row of `df` and return df + 6 new columns."""
    client  = _make_client()
    texts   = df["review_text"].tolist()
    batches = [texts[i:i + LLM_BATCH_SIZE] for i in range(0, len(texts), LLM_BATCH_SIZE)]
    est_min = len(batches) * RATE_LIMIT_SLEEP / 60

    print(f"Extracting LLM features: {len(texts):,} reviews | {len(batches)} batches | ~{est_min:.0f} min")
    print(f"  Model: {LLM_MODEL} | sleep: {RATE_LIMIT_SLEEP}s/batch\n")

    all_signals: list[dict] = []
    failure_count = 0

    for batch_idx, batch in enumerate(tqdm(batches, unit="batch")):
        try:
            result = _extract_batch_typed(client, batch)
            signals = _align_batch(result.signals, len(batch))
        except Exception as exc:  # noqa: BLE001, any failure degrades the batch to neutral
            # `Exception` already covers tenacity's RetryError; the retry decorator
            # also re-raises the underlying error, so RetryError is never seen here.
            tqdm.write(f"  Batch {batch_idx} failed after {MAX_RETRIES} retries: {exc}")
            signals = [NEUTRAL_SIGNAL] * len(batch)
            failure_count += 1

        all_signals.extend([s.model_dump() for s in signals])
        if batch_idx < len(batches) - 1:
            time.sleep(RATE_LIMIT_SLEEP)

    if failure_count:
        print(f"\n  Warning: {failure_count}/{len(batches)} batches used neutral fallback")

    features_df = pd.DataFrame(all_signals, index=df.index)
    for col in ("technical_issue", "value_complaint", "engagement_dropped", "positive_signal"):
        features_df[col] = features_df[col].astype(int)

    result_df = pd.concat([df, features_df], axis=1)
    if save:
        result_df.to_parquet(FEATURES_PATH, index=False)
        print(f"  Saved → {FEATURES_PATH}")
    return result_df


def extract_single(text: str) -> dict:
    """Extract signals for a single review, used by the Streamlit live tab."""
    client = _make_client()
    result = _extract_batch_typed(client, [text])
    return _align_batch(result.signals, 1)[0].model_dump()
