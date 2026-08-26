"""BootstrapFewShot the segmenter against both candidate models, using the
Gemini teacher to generate demonstrations (self-bootstrapping doesn't work
here — Qwen3-4B's zero-shot pass rate is ~0, so it can't harvest its own
correct traces). Evaluates compiled vs. zero-shot baseline on a held-out
set for both models.

Run with: .venv/bin/python scripts/optimize_segmenter.py
"""

import json
import os
import dspy
from dotenv import load_dotenv
from dspy.teleprompt import BootstrapFewShot

from earnings_analyser.analysis.segmenter_metric import adjusted_rand_index
from earnings_analyser.signatures.segmenter import Segmenter

load_dotenv()

TEACHER_LM = dspy.LM("gemini/gemini-flash-lite-latest", api_key=os.environ["GOOGLE_API_KEY"].strip(), temperature=0.0)
STUDENTS = {
    "qwen3:4b-instruct": "ollama_chat/qwen3:4b-instruct",
    "qwen2.5:7b-instruct": "ollama_chat/qwen2.5:7b-instruct-q4_K_M",
}


def load_examples(path: str) -> list[dspy.Example]:
    scenarios = json.loads(Path(path).read_text())
    examples = []
    for s in scenarios:
        items = [f"[{i}] {text}" for i, text in enumerate(s["items"])]
        ex = dspy.Example(items=items, max_clusters=s["max_clusters"], truth=s["truth"])
        examples.append(ex.with_inputs("items", "max_clusters"))
    return examples


def metric(example, pred, trace=None) -> float:
    try:
        return adjusted_rand_index(pred.cluster_id, example.truth)
    except Exception:
        return 0.0


def evaluate(program, examples: list[dspy.Example]) -> float:
    scores = []
    for ex in examples:
        try:
            pred = program(items=ex["items"], max_clusters=ex.max_clusters)
            scores.append(adjusted_rand_index(pred.cluster_id, ex.truth))
        except Exception:
            scores.append(0.0)
    return sum(scores) / len(scores)


def main() -> None:
    train = load_examples("/tmp/segmenter_train.json")
    heldout = load_examples("/tmp/segmenter_heldout.json")
    print(f"train={len(train)} heldout={len(heldout)}\n")

    for name, model_id in STUDENTS.items():
        student_lm = dspy.LM(model_id, api_base="http://localhost:11434", api_key="ollama", temperature=0.0)
        dspy.configure(lm=student_lm)

        baseline = dspy.Predict(Segmenter)
        baseline_score = evaluate(baseline, heldout)
        print(f"[{name}] zero-shot held-out ARI: {baseline_score:.3f}")

        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
            teacher_settings=dict(lm=TEACHER_LM),
        )
        compiled = optimizer.compile(student=dspy.Predict(Segmenter), trainset=train)

        compiled_score = evaluate(compiled, heldout)
        print(f"[{name}] compiled  held-out ARI: {compiled_score:.3f}")
        print(f"[{name}] delta: {compiled_score - baseline_score:+.3f}\n")


if __name__ == "__main__":
    main()
