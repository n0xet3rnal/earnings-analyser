"""Run the combined DimensionScores signature against real transcript
sections (prepared remarks + Q&A), and verify temperature=0 gives
reproducible output on repeated calls.

Run with: .venv/bin/python scripts/test_signatures.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dspy
from datasets import load_dataset

from earnings_analyser.config import DEFAULT_MODEL, configure_dspy
from earnings_analyser.data.splitter import regex_split_transcript as split_transcript
from earnings_analyser.signatures import DIMENSIONS, DimensionScores, score_of


def print_result(result) -> None:
    for dim in DIMENSIONS:
        label = getattr(result, dim)
        rationale = getattr(result, f"{dim}_rationale")
        print(f"  [{dim}] {label} ({score_of(label):+.1f}) — {rationale}")


def main() -> None:
    configure_dspy()

    ds = load_dataset("glopardo/sp500-earnings-transcripts")["train"]
    row = ds[0]
    split = split_transcript(row["transcript"])
    print(f"Testing on {row['ticker']} {row['earnings_date']}, model={DEFAULT_MODEL}\n")

    predictor = dspy.Predict(DimensionScores)

    print("=== Prepared remarks ===")
    r_prepared = predictor(section_text=split.prepared_remarks, section_type="prepared_remarks")
    print_result(r_prepared)

    print("\n=== Q&A ===")
    r_qa = predictor(section_text=split.qa_section, section_type="qa")
    print_result(r_qa)

    print("\n--- Determinism check: Q&A section run twice, cache disabled ---")
    uncached_lm = dspy.LM(
        DEFAULT_MODEL, api_key=os.environ["GEMINI_API_KEY"], temperature=0.0, cache=False
    )
    with dspy.context(lm=uncached_lm):
        r1 = predictor(section_text=split.qa_section, section_type="qa")
        r2 = predictor(section_text=split.qa_section, section_type="qa")
    labels1 = [getattr(r1, d) for d in DIMENSIONS]
    labels2 = [getattr(r2, d) for d in DIMENSIONS]
    print(f"run 1: {labels1}")
    print(f"run 2: {labels2}")
    print("MATCH" if labels1 == labels2 else "MISMATCH — temperature=0 did not give reproducible output")


if __name__ == "__main__":
    main()
