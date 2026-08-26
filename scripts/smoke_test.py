"""Sanity check: DSPy can reach the configured local Ollama endpoint.

Run with: .venv/bin/python scripts/smoke_test.py
Requires `ollama serve` running locally with the configured model pulled
(see `src/earnings_analyser/config.py`'s DEFAULT_MODEL).
"""

import dspy

from earnings_analyser.config import configure_dspy


class Ping(dspy.Signature):
    """Reply with a single short sentence confirming you are online."""

    note: str = dspy.OutputField()


def main() -> None:
    configure_dspy()
    predictor = dspy.Predict(Ping)
    result = predictor()
    print("Local model responded:", result.note)


if __name__ == "__main__":
    main()
