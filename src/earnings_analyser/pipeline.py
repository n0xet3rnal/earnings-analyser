"""End-to-end orchestration: sentence-split -> recursive collapse ->
graph store. Call `configure_dspy()` (see `config.py`) before this.

Kept deliberately thin — `modules/collapse_step.py` owns the actual
recursion, this just wires live predictors to it against a given
transcript and graph store.
"""

import dspy

from .modules.collapse_step import (
    DEFAULT_BRANCHING_FACTOR,
    DEFAULT_MAX_WORKERS,
    DEFAULT_TARGET_TERMINAL_COUNT,
    collapse,
)
from .persistence.graph_store import GraphStore
from .signatures import DimensionCollapse, DimensionCollapseBundled


def run_pipeline(
    store: GraphStore,
    transcript_text: str,
    window_size: int = 8,
    window_overlap: int = 0,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    target: int = DEFAULT_TARGET_TERMINAL_COUNT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    predictor = dspy.Predict(DimensionCollapse)
    bundled_predictor = dspy.Predict(DimensionCollapseBundled)
    collapse(
        store,
        predictor,
        bundled_predictor,
        transcript_text,
        window_size=window_size,
        window_overlap=window_overlap,
        branching_factor=branching_factor,
        target=target,
        max_workers=max_workers,
    )
