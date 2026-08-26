"""Compact-prompt predictor for the collapse calls, bypassing DSPy's
default adapter (measured ~43% pure scaffolding, not model content —
see implementation-plan.md). Backend-agnostic: takes any `dspy.LM`
instance (Ollama or a cloud provider like Gemini) and calls it directly,
never through `dspy.Predict`/Signature/Adapter.
"""

from typing import Any

import dspy

from ..signatures.common import DIMENSIONS
from ..signatures.dimension_collapse import _RUBRIC

_FORMAT_SPEC = """
Respond with EXACTLY 6 lines, one per dimension, IN THIS ORDER: forward_guidance, uncertainty, confidence, sentiment, macro_focus, jargon.
Each line format (pipe-delimited, no spaces around pipes):
<weights>|<one-sentence summary, 25 words max>|<label 1-5 or blank if not terminal>
"""


class _DimResult:
    def __init__(self, per_dim: dict[str, dict[str, Any]]):
        self._per_dim = per_dim

    def __getattr__(self, name: str):
        for dim in DIMENSIONS:
            for suffix, key in ((f"{dim}_weights", "weights"), (f"{dim}_summary", "summary"), (f"{dim}_label", "label")):
                if name == suffix:
                    return self._per_dim.get(dim, {}).get(key)
        raise AttributeError(name)


def _call_lm(lm: dspy.LM, prompt: str) -> str:
    """Direct call through the configured dspy.LM — whatever backend it
    wraps (Ollama, Gemini, anything litellm supports) — never through
    dspy.Predict/Signature/Adapter, which is what carries the scaffolding
    overhead."""
    response = lm(messages=[{"role": "user", "content": prompt}])
    return response[0] if isinstance(response, list) else response


def _parse_compact(text: str) -> dict[str, dict[str, Any]]:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    out: dict[str, dict[str, Any]] = {}
    for dim, line in zip(DIMENSIONS, lines):
        parts = line.split("|")
        weights_raw, summary, label_raw = (parts + ["", "", ""])[:3]
        try:
            weights = [int(x) for x in weights_raw.split(",") if x.strip()]
        except ValueError:
            weights = []
        label = int(label_raw) if label_raw.strip().lstrip("-").isdigit() else None
        out[dim] = {"weights": weights, "summary": summary.strip(), "label": label}
    for dim in DIMENSIONS:
        out.setdefault(dim, {"weights": [], "summary": "", "label": None})
    return out


class RawBasePredictor:
    """Base-level (window) call: all six dimensions read the same shared
    item list — safe to show it once, not once per dimension."""

    def __init__(self, lm: dspy.LM | None = None):
        self.lm = lm  # None -> use dspy.settings.lm at call time

    def __call__(self, group_items: list[str], is_terminal: bool) -> _DimResult:
        n = len(group_items)
        items_block = "\n".join(group_items)
        prompt = f"""{_RUBRIC}
{_FORMAT_SPEC}
The weight count is NOT six (six is the number of dimensions/lines, a
different thing) — it equals the number of INPUTS below. Worked example
for a batch of 2 inputs (2 inputs -> 2 weights, one line shown):
2,0|Example summary sentence here, not real content.|3

This batch has {n} inputs, numbered [0] to [{n - 1}]. Every weight list
you write below must have EXACTLY {n} numbers, comma separated, one per
input in order — not six, {n}.

Inputs (is_terminal={is_terminal}):
{items_block}

Reminder: {n} inputs above -> {n} weights per line, all 6 lines.
"""
        lm = self.lm or dspy.settings.lm
        text = _call_lm(lm, prompt)
        return _DimResult(_parse_compact(text))


class RawBundledPredictor:
    """Collapse-round call: each dimension reads its OWN item list, since
    composites have already diverged per dimension by this level."""

    def __init__(self, lm: dspy.LM | None = None):
        self.lm = lm

    def __call__(self, **kwargs: Any) -> _DimResult:
        is_terminal = kwargs["is_terminal"]
        sections = []
        for dim in DIMENSIONS:
            items = kwargs[f"{dim}_inputs"]
            block = "\n".join(items)
            sections.append(f"{dim} inputs ({len(items)} total, indexed [0]-[{len(items) - 1}]):\n{block}")
        items_block = "\n\n".join(sections)

        prompt = f"""{_RUBRIC}
{_FORMAT_SPEC}
Each dimension has its OWN separate input list below — do not mix
content between them. Each line's weight count must match THAT
dimension's own input count shown above its list, not a fixed number
and not six. Worked example (a dimension with 2 inputs -> 2 weights):
2,0|Example summary sentence here, not real content.|3

Inputs (is_terminal={is_terminal}):
{items_block}
"""
        lm = self.lm or dspy.settings.lm
        text = _call_lm(lm, prompt)
        return _DimResult(_parse_compact(text))
