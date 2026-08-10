"""End-to-end: transcript -> pipeline -> composite score.

Run with: .venv/bin/python scripts/test_composite.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset

from earnings_analyser.composite import compute_composite
from earnings_analyser.config import configure_dspy
from earnings_analyser.pipeline import TranscriptScorer


def main() -> None:
    configure_dspy()
    ds = load_dataset("glopardo/sp500-earnings-transcripts")["train"]
    row = ds[0]

    scorer = TranscriptScorer(n_samples=3)
    scores = scorer(ticker=row["ticker"], earnings_date=row["earnings_date"], transcript=row["transcript"])
    result = compute_composite(scores)

    print(f"{scores.ticker} {scores.earnings_date}")
    print(f"Composite score: {result.composite_score:+.3f}  (-1 bearish .. +1 bullish)\n")
    print("Per-section (bullish-signed):")
    for name, score in result.section_scores.items():
        print(f"  {name}: {score:+.3f}")
    print("\nPer-dimension (bullish-signed, blended across sections):")
    for dim, score in result.dimension_contributions.items():
        print(f"  {dim}: {score:+.3f}")


if __name__ == "__main__":
    main()
