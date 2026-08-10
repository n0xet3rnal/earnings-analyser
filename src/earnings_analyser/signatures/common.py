"""Shared scoring scale and dimension metadata.

All 6 dimensions classify into one of 5 discrete levels rather than a raw
float — LLMs are reliable classifiers but poor at inventing calibrated
numbers on the fly, and inconsistent across repeated runs. Combined with
temperature=0 in the LM config, this keeps scores reproducible.

Each dimension is scored in its own natural direction (e.g. Uncertainty:
+1.0 means MORE uncertainty, not less). AGGREGATION_SIGN records whether a
high raw score is bullish (+1) or bearish (-1) for the composite — used by
aggregation (Phase 4), not by the signature itself.
"""

from typing import Literal

ScoreLabel = Literal["strong_negative", "mild_negative", "neutral", "mild_positive", "strong_positive"]

ANCHOR_SCORES: dict[str, float] = {
    "strong_negative": -1.0,
    "mild_negative": -0.3,
    "neutral": 0.0,
    "mild_positive": 0.3,
    "strong_positive": 1.0,
}

DIMENSIONS = ["forward_guidance", "uncertainty", "confidence", "sentiment", "macro_focus", "jargon"]

# Bullish-natural dimensions (+1 raw score = favorable) vs. magnitude-natural
# dimensions (+1 raw score = more of a trait that's a mild-to-strong yellow
# flag). Uncertainty is magnitude-natural and gets upweighted per Matera
# 2024's finding that analysts underweight it.
AGGREGATION_SIGN: dict[str, int] = {
    "forward_guidance": 1,
    "uncertainty": -1,
    "confidence": 1,
    "sentiment": 1,
    "macro_focus": -1,
    "jargon": -1,
}


def score_of(label: str) -> float:
    return ANCHOR_SCORES[label]
