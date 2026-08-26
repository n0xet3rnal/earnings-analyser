"""Shared label scale and dimension list."""

from typing import Literal

ScoreLabel = Literal["strong_negative", "mild_negative", "neutral", "mild_positive", "strong_positive"]

# Index 1-5 -> word. The model outputs the int (see dimension_collapse.py);
# no word vocabulary at that boundary means nothing to hallucinate/leak.
LABEL_WORDS: list[str] = ["strong_negative", "mild_negative", "neutral", "mild_positive", "strong_positive"]

DIMENSIONS = ["forward_guidance", "uncertainty", "confidence", "sentiment", "macro_focus", "jargon"]
