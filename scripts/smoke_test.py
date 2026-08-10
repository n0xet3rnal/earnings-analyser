"""Sanity check: DSPy can reach Gemini Flash with the configured API key.

Run with: .venv/bin/python scripts/smoke_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dspy

from earnings_analyser.config import configure_dspy


class Ping(dspy.Signature):
    """Reply with a single short sentence confirming you are online."""

    note: str = dspy.OutputField()


def main() -> None:
    configure_dspy()
    predictor = dspy.Predict(Ping)
    result = predictor()
    print("Gemini Flash responded:", result.note)


if __name__ == "__main__":
    main()
