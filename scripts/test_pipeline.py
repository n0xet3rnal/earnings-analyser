"""End-to-end pipeline test: raw transcript -> split -> 6-dim scores for
both sections, using self-consistency (3 samples/section).

Run with: .venv/bin/python scripts/test_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from datasets import load_dataset

from earnings_analyser.config import configure_dspy
from earnings_analyser.pipeline import TranscriptScorer
from earnings_analyser.signatures import DIMENSIONS


def print_section(name: str, section) -> None:
    print(f"=== {name} ===")
    if section is None:
        print("  (no section detected)")
        return
    for dim in DIMENSIONS:
        r = section.dimensions[dim]
        print(f"  [{dim}] {r.label} score={r.score:+.2f} agreement={r.agreement:.0%}")
        print(f"      {r.rationale}")


def main() -> None:
    configure_dspy()
    ds = load_dataset("glopardo/sp500-earnings-transcripts")["train"]
    row = ds[0]

    scorer = TranscriptScorer(n_samples=3)
    result = scorer(ticker=row["ticker"], earnings_date=row["earnings_date"], transcript=row["transcript"])

    print(f"{result.ticker} {result.earnings_date} (split_confidence={result.split_confidence})\n")
    print_section("Prepared Remarks", result.prepared_remarks)
    print()
    print_section("Q&A", result.qa)


if __name__ == "__main__":
    main()
