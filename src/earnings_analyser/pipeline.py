"""End-to-end orchestration: sentence-split -> recursive collapse ->
graph store.

Kept deliberately thin — `modules/collapse_step.py` owns the actual
recursion, this just wires live predictors to it against a given
transcript and graph store. Predictors are `modules/raw_collapse_predictor.py`'s
compact-prompt implementations (bypasses DSPy's default adapter — measured
~2.5-2.9x faster, same model, same task), backend-agnostic: pass any
configured `dspy.LM` (local Ollama or a cloud provider).
"""

import dspy

from .modules.collapse_step import (
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TARGET_TERMINAL_COUNT,
    collapse,
)
from .modules.raw_collapse_predictor import RawBasePredictor, RawBundledPredictor
from .persistence.graph_store import GraphStore


def run_pipeline(
    store: GraphStore,
    transcript_text: str,
    window_size: int = 8,
    window_overlap: int = 0,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    target: int = DEFAULT_TARGET_TERMINAL_COUNT,
    max_workers: int = DEFAULT_MAX_WORKERS,
    requests_per_minute: int | None = None,
    lm: dspy.LM | None = None,
) -> None:
    collapse(
        store,
        RawBasePredictor(lm=lm),
        RawBundledPredictor(lm=lm),
        transcript_text,
        window_size=window_size,
        window_overlap=window_overlap,
        branching_factor=branching_factor,
        target=target,
        max_workers=max_workers,
        requests_per_minute=requests_per_minute,
    )
