"""One level of the recursive weighted collapse (implementation-plan.md §2).

`collapse()` is the whole orchestration: sentence-split the transcript,
run the base (window) level, then keep collapsing per-dimension composite
pools by a fixed branching factor per round until each dimension is down
to `target` terminal composites. Every level's weights get written to the
graph store as edges; `analysis/attribution.py` is what turns those into
per-sentence importance scores later — this module only ever writes them.

The base level (level 1) is dimension-agnostic: one call per window,
using `DimensionCollapse`, covers all six dimensions against the same
shared raw sentences — safe to bundle because it's one text span read
once. From level 2 onward, composites have already diverged per
dimension (a "Uncertainty" composite is different text from a
"Confidence" composite of the same window), so `DimensionCollapseBundled`
is used instead: one call per group still covers all six dimensions, but
each dimension gets its own separate input list rather than sharing one.
Composite counts start identical across dimensions and a fixed branching
factor keeps them in lockstep every round after that — one shared round
count applies to all six dimensions rather than tracking each
separately, which is also what makes "one call per group index" possible
at level 2+: batch `i` is the same size for every dimension.

Calls within a level are independent (no group's call depends on another
group's output), so both levels dispatch their calls through a thread
pool and only write to the graph store afterward, sequentially, from the
main thread — this sidesteps SQLite's not-thread-safe-by-default
connections entirely rather than adding locking.
"""

import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from ..data.sentence_split import Window, base_windows, split_sentences
from ..persistence.graph_store import Edge, GraphStore, Node
from ..signatures import DIMENSIONS, MAX_RELEVANCE_SCORE
from ..signatures.common import LABEL_WORDS

DEFAULT_BRANCHING_FACTOR = 16
DEFAULT_TARGET_TERMINAL_COUNT = 5
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_WORKERS = 4


class CollapsePredictor(Protocol):
    def __call__(self, **kwargs: Any): ...


def _format_items(texts: list[str]) -> list[str]:
    return [f"[{i}] {text}" for i, text in enumerate(texts)]


def _predict_with_retry(predictor: CollapsePredictor, kwargs: dict[str, Any],
                         max_attempts: int = DEFAULT_MAX_ATTEMPTS):
    """A local model occasionally returns malformed structured output
    (e.g. reaching for the wrong enum's vocabulary on one field). One bad
    call shouldn't take down an otherwise-successful run — retry a few
    times before giving up."""
    last_error: Exception | None = None
    for _ in range(max_attempts):
        try:
            return predictor(**kwargs)
        except Exception as exc:  # dspy/pydantic validation errors, transient backend errors
            last_error = exc
    raise RuntimeError(f"predictor failed after {max_attempts} attempts") from last_error


def _run_concurrently(predictor: CollapsePredictor, calls: list[dict[str, Any]], max_workers: int):
    """Dispatch one `_predict_with_retry` call per entry in `calls`
    through a thread pool — I/O-bound waiting on Ollama's HTTP response,
    not CPU-bound, so threads (not processes) are the right stdlib tool
    here. Returns results in the same order as `calls`."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_predict_with_retry, predictor, kwargs) for kwargs in calls]
        return [f.result() for f in futures]


def _normalized_weights(raw: list, n_items: int) -> list[float]:
    """Clamp each item's 0-3 relevance score into range, then normalize
    across this group's own inputs so they sum to 1 — the invariant the
    attribution matrix chain in `analysis/attribution.py` depends on. A
    short/long/unparseable list (the model didn't return exactly
    `n_items` entries) is padded or truncated rather than raising —
    same "degrade quietly" principle as the label fallback."""
    def _score(v) -> float:
        try:
            return max(0, min(MAX_RELEVANCE_SCORE, int(v)))
        except (TypeError, ValueError):
            return 0.0

    padded = list(raw)[:n_items] + [0] * max(0, n_items - len(raw))
    scores = [_score(v) for v in padded]
    total = sum(scores)
    if total <= 0:
        return [0.0] * n_items
    return [s / total for s in scores]


def _label_word(raw: int | None) -> str | None:
    """1-5 -> word (LABEL_WORDS), clamped. Any non-int/out-of-range value
    (including the model writing "null" as text) becomes no label rather
    than a crash or a stray literal string reaching the UI."""
    try:
        index = int(raw)
    except (TypeError, ValueError):
        return None
    return LABEL_WORDS[max(1, min(len(LABEL_WORDS), index)) - 1]


def _write_dimension_result(store: GraphStore, result, dim: str, node_id: str, level: int,
                             child_ids: list[str], n_items: int, is_terminal: bool) -> None:
    """Shared by both levels: pull one dimension's fields off a
    (possibly multi-dimension) result object and write the composite
    node + its edges."""
    weights = getattr(result, f"{dim}_weights")
    summary = getattr(result, f"{dim}_summary")
    label = _label_word(getattr(result, f"{dim}_label")) if is_terminal else None
    normalized = _normalized_weights(weights, n_items)

    store.add_nodes(
        [Node(node_id=node_id, level=level, kind="composite", text=summary, dimension=dim,
              label=label, terminal=is_terminal)]
    )
    store.add_edges(
        [
            Edge(child=child_id, parent=node_id, dimension=dim, weight=w)
            for child_id, w in zip(child_ids, normalized)
            if w > 0
        ]
    )


def run_base_level(
    store: GraphStore,
    predictor: CollapsePredictor,
    transcript_text: str,
    window_size: int,
    window_overlap: int,
    target: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> tuple[int, int]:
    """Sentence-split, group into overlapping base windows, run one
    `DimensionCollapse` call per window (all six dimensions at once —
    see module docstring), concurrently. Returns
    (level, composite_count_per_dimension) for the caller's collapse loop."""
    sentences = split_sentences(transcript_text)
    store.add_nodes(
        [
            Node(node_id=f"s{s.index}", level=0, kind="source_sentence", text=s.text, start=s.start, end=s.end)
            for s in sentences
        ]
    )
    windows: list[Window] = base_windows(sentences, size=window_size, overlap=window_overlap)
    is_terminal = len(windows) <= target

    calls = [
        {"group_items": _format_items([s.text for s in w.sentences]), "is_terminal": is_terminal}
        for w in windows
    ]
    results = _run_concurrently(predictor, calls, max_workers)

    for window, result in zip(windows, results):
        child_ids = [f"s{s.index}" for s in window.sentences]
        for dim in DIMENSIONS:
            _write_dimension_result(
                store, result, dim, node_id=f"w{window.index}-{dim}", level=1,
                child_ids=child_ids, n_items=len(child_ids), is_terminal=is_terminal,
            )

    store.mark_level_complete(1)
    return 1, len(windows)


def run_collapse_round(
    store: GraphStore,
    bundled_predictor: CollapsePredictor,
    level: int,
    branching_factor: int,
    target: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> int:
    """One more round: group the previous level's composites per
    dimension into batches of `branching_factor`. Batch `i` is the same
    size for every dimension (lockstep — see module docstring), so one
    `DimensionCollapseBundled` call per group index covers all six
    dimensions' batch `i` at once, instead of one call per
    (dimension, batch) — a 6x reduction in call count at this level.
    Strict partition — every composite belongs to exactly one group here,
    unlike the base level's overlapping windows (see
    implementation-plan.md §2.3: composites are already synthesized,
    coherent text, not raw fragments that can be arbitrarily split)."""
    new_level = level + 1

    per_dim_batches: dict[str, list[tuple]] = {
        dim: list(itertools.batched(
            [n for n in store.nodes_at_level(level, kind="composite") if n.dimension == dim],
            branching_factor,
        ))
        for dim in DIMENSIONS
    }
    n_groups = len(per_dim_batches[DIMENSIONS[0]])  # identical across dimensions, by construction
    is_terminal = n_groups <= target

    calls = [
        {
            **{f"{dim}_inputs": _format_items([n.text for n in per_dim_batches[dim][i]]) for dim in DIMENSIONS},
            "is_terminal": is_terminal,
        }
        for i in range(n_groups)
    ]
    results = _run_concurrently(bundled_predictor, calls, max_workers)

    for group_index, result in enumerate(results):
        for dim in DIMENSIONS:
            batch = per_dim_batches[dim][group_index]
            _write_dimension_result(
                store, result, dim, node_id=f"L{new_level}-{dim}-{group_index}", level=new_level,
                child_ids=[n.node_id for n in batch], n_items=len(batch), is_terminal=is_terminal,
            )

    store.mark_level_complete(new_level)
    return n_groups


def collapse(
    store: GraphStore,
    predictor: CollapsePredictor,
    bundled_predictor: CollapsePredictor,
    transcript_text: str,
    window_size: int = 8,
    window_overlap: int = 0,
    branching_factor: int = DEFAULT_BRANCHING_FACTOR,
    target: int = DEFAULT_TARGET_TERMINAL_COUNT,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> None:
    """Full orchestration: base level (`DimensionCollapse`), then
    collapse rounds (`DimensionCollapseBundled`) until each dimension is
    down to `target` terminal composites. Resumable — skips straight to
    the next level if `store.last_complete_level()` shows earlier levels
    already finished."""
    resume_from = store.last_complete_level()

    if resume_from is None:
        level, count = run_base_level(
            store, predictor, transcript_text, window_size, window_overlap, target, max_workers
        )
    else:
        level = resume_from
        count = len([n for n in store.nodes_at_level(level, kind="composite") if n.dimension == DIMENSIONS[0]])

    while count > target:
        count = run_collapse_round(store, bundled_predictor, level, branching_factor, target, max_workers)
        level += 1
