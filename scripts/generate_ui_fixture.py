"""Run the real pipeline once against a production-size sample transcript
(`fixtures/sample_transcript.txt`) and save the resulting graph store as a
fixture — lets the "Test with sample data" button in app.py replay a
real, full-size run without spending LLM calls every time the UI is
tweaked.

Run with: .venv/bin/python scripts/generate_ui_fixture.py
Needs a working backend (see config.py) — this is the one time it's
actually called. Full-size transcript means this makes real production
volumes of calls against that backend.
"""

from pathlib import Path

from earnings_analyser.config import configure_dspy
from earnings_analyser.persistence.graph_store import GraphStore
from earnings_analyser.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
FIXTURE_PATH = ROOT / "fixtures" / "sample_run.sqlite"
TRANSCRIPT_PATH = ROOT / "fixtures" / "sample_transcript.txt"


def main() -> None:
    lm, pipeline_profile, backend = configure_dspy()
    transcript_text = TRANSCRIPT_PATH.read_text()
    print(f"Generating fixture with backend={backend!r}, {len(transcript_text.split())} words ...")

    FIXTURE_PATH.parent.mkdir(exist_ok=True)
    FIXTURE_PATH.unlink(missing_ok=True)

    with GraphStore(FIXTURE_PATH) as store:
        run_pipeline(store, transcript_text, lm=lm, **pipeline_profile)

    print(f"Saved fixture to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
