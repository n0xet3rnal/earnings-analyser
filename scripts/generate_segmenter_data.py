"""Generate synthetic segmenter training/held-out data with a Gemini
teacher (implementation plan: DSPy-optimize a segmenter). Each scenario:
`num_threads` threads, each internally connected by a *named*
relationship type, not by repeated vocabulary. Ground truth (which
thread each sentence came from) is known by construction.

Run with: .venv/bin/python scripts/generate_segmenter_data.py --n 8 --out /tmp/segmenter_pilot.json
"""

import argparse
import json
import random
import dspy
from dotenv import load_dotenv

RELATIONSHIP_TYPES = ["causal", "contrastive", "elaborative"]  # "rhetorical" dropped: 2/2 pilot instances read as loose topical bucketing, not real setup-payoff

RELATIONSHIP_DESC = {
    "causal": "each later sentence explains WHY an earlier one is true, using different vocabulary than the claim itself (e.g. a result, then its unstated-in-words cause)",
    "rhetorical": "a setup sentence (implicit question or framing) followed by its payoff/resolution, phrased without repeating the setup's words",
    "contrastive": "sentences that push back against or complicate an earlier one in the same thread, using contrast rather than restatement",
    "elaborative": "sentences that refer back to or build on an earlier one anaphorically ('that', 'this', 'as mentioned') without repeating its keywords",
}


class GenerateScenario(dspy.Signature):
    """Write synthetic earnings-call-register sentences for testing thematic
    clustering, NOT topic-word matching. Produce exactly `num_threads`
    threads, each with exactly `sentences_per_thread` sentences. Every
    thread must be internally connected via the given relationship type
    (see its description) — deliberately avoid repeating the same keyword
    across a thread's sentences; the connection must require understanding
    what's being said, not spotting shared vocabulary. Different threads
    must be about genuinely different underlying topics."""

    num_threads: int = dspy.InputField()
    sentences_per_thread: int = dspy.InputField()
    relationship_type: str = dspy.InputField()
    relationship_description: str = dspy.InputField()

    threads: list[list[str]] = dspy.OutputField(
        desc="Exactly num_threads lists, each with exactly sentences_per_thread sentences"
    )


def make_scenario(predictor: dspy.Predict, num_threads: int, sentences_per_thread: int,
                   relationship_type: str, rng: random.Random) -> dict | None:
    result = predictor(
        num_threads=num_threads,
        sentences_per_thread=sentences_per_thread,
        relationship_type=relationship_type,
        relationship_description=RELATIONSHIP_DESC[relationship_type],
    )
    threads = result.threads

    # Quality gate: cheap, code-level shape check only (no second-teacher
    # judge pass — skipped for now, add if bad scenarios turn out common
    # in spot review of the pilot batch).
    if len(threads) != num_threads or any(len(t) != sentences_per_thread for t in threads):
        return None

    sentences, truth = [], []
    for thread_id, thread in enumerate(threads):
        for s in thread:
            sentences.append(s)
            truth.append(thread_id)

    order = list(range(len(sentences)))
    rng.shuffle(order)
    return {
        "items": [sentences[i] for i in order],
        "truth": [truth[i] for i in order],
        "max_clusters": num_threads,
        "relationship_type": relationship_type,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    load_dotenv()
    import os
    lm = dspy.LM("gemini/gemini-flash-lite-latest", api_key=os.environ["GOOGLE_API_KEY"].strip(), temperature=0.7)
    dspy.configure(lm=lm)
    predictor = dspy.Predict(GenerateScenario)

    rng = random.Random(args.seed)
    scenarios = []
    attempts = 0
    while len(scenarios) < args.n and attempts < args.n * 3:
        attempts += 1
        num_threads = rng.choice([2, 3, 4])
        sentences_per_thread = rng.choice([3, 4, 5])
        relationship_type = RELATIONSHIP_TYPES[attempts % len(RELATIONSHIP_TYPES)]
        try:
            scenario = make_scenario(predictor, num_threads, sentences_per_thread, relationship_type, rng)
        except Exception as exc:
            print(f"  attempt {attempts} failed: {exc}")
            continue
        if scenario is None:
            print(f"  attempt {attempts} rejected: shape mismatch")
            continue
        scenarios.append(scenario)
        print(f"  [{len(scenarios)}/{args.n}] {relationship_type}, {num_threads} threads x {sentences_per_thread}")

    Path(args.out).write_text(json.dumps(scenarios, indent=2))
    print(f"wrote {len(scenarios)} scenarios to {args.out}")


if __name__ == "__main__":
    main()
