import threading

import pytest

from earnings_analyser.analysis.attribution import compute_attribution
from earnings_analyser.modules.collapse_step import RateLimiter, _predict_with_retry, collapse
from earnings_analyser.persistence.graph_store import GraphStore
from earnings_analyser.signatures import DIMENSIONS


class _FakePrediction:
    """Stands in for a DimensionCollapse/DimensionCollapseBundled
    dspy.Predict result: every input gets a relevance score of 3
    (uniform after normalization) and, when terminal, labeled 'neutral'
    — enough to exercise the orchestration and graph structure without a
    live model."""

    def __init__(self, n_items_by_dim: dict[str, int], is_terminal: bool):
        self._n_items_by_dim = n_items_by_dim
        self._is_terminal = is_terminal

    def __getattr__(self, name: str):
        for dim in DIMENSIONS:
            if name == f"{dim}_weights":
                return [3] * self._n_items_by_dim[dim]
            if name == f"{dim}_summary":
                return f"summary for {dim} over {self._n_items_by_dim[dim]} items"
            if name == f"{dim}_label":
                return 3 if self._is_terminal else None
        raise AttributeError(name)


def _fake_predictor(**kwargs) -> _FakePrediction:
    """Handles both call shapes: base-level `group_items` (shared across
    dimensions) and collapse-round `{dim}_inputs` (one list per
    dimension) — same fake works as both `predictor` and
    `bundled_predictor` in `collapse()`."""
    is_terminal = kwargs["is_terminal"]
    if "group_items" in kwargs:
        n = len(kwargs["group_items"])
        n_items_by_dim = {dim: n for dim in DIMENSIONS}
    else:
        n_items_by_dim = {dim: len(kwargs[f"{dim}_inputs"]) for dim in DIMENSIONS}
    return _FakePrediction(n_items_by_dim, is_terminal)


def _synthetic_transcript(n_sentences: int) -> str:
    return " ".join(f"This is sentence number {i}." for i in range(n_sentences))


def test_collapse_converges_and_stays_in_lockstep_across_dimensions(tmp_path):
    transcript = _synthetic_transcript(30)

    with GraphStore(tmp_path / "graph.sqlite") as store:
        collapse(
            store,
            _fake_predictor,
            _fake_predictor,
            transcript,
            window_size=3,
            window_overlap=1,
            branching_factor=2,
            target=2,
        )

        assert store.last_complete_level() is not None

        terminal_counts = {dim: len(store.terminal_nodes(dim)) for dim in DIMENSIONS}
        # lockstep property (module docstring): identical counts across dimensions
        assert len(set(terminal_counts.values())) == 1
        assert all(count <= 2 for count in terminal_counts.values())
        assert all(count >= 1 for count in terminal_counts.values())


def test_every_terminal_attribution_sums_to_one(tmp_path):
    transcript = _synthetic_transcript(20)

    with GraphStore(tmp_path / "graph.sqlite") as store:
        collapse(store, _fake_predictor, _fake_predictor, transcript,
                 window_size=4, window_overlap=1, branching_factor=3, target=3)

        for dim in DIMENSIONS:
            attribution = compute_attribution(store, dim)
            for terminal_id, leaf_weights in attribution.items():
                assert sum(leaf_weights.values()) == pytest.approx(1.0)


def test_resuming_skips_already_completed_levels(tmp_path):
    transcript = _synthetic_transcript(20)
    db_path = tmp_path / "graph.sqlite"

    calls = {"n": 0}
    lock = threading.Lock()

    def counting_predictor(**kwargs):
        # collapse() dispatches calls through a thread pool now — a bare
        # += here would be a real race, not just a style nit.
        with lock:
            calls["n"] += 1
        return _fake_predictor(**kwargs)

    with GraphStore(db_path) as store:
        collapse(store, counting_predictor, counting_predictor, transcript,
                 window_size=4, window_overlap=1, branching_factor=3, target=3)
    first_run_calls = calls["n"]

    # Re-running against the same store should resume from the last
    # completed level rather than redoing everything from scratch.
    with GraphStore(db_path) as store:
        collapse(store, counting_predictor, counting_predictor, transcript,
                 window_size=4, window_overlap=1, branching_factor=3, target=3)

    assert calls["n"] == first_run_calls


def test_predict_with_retry_backs_off_between_attempts(monkeypatch):
    sleeps = []
    monkeypatch.setattr("earnings_analyser.modules.collapse_step.time.sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr("earnings_analyser.modules.collapse_step.random.uniform", lambda a, b: 0.0)

    attempts = {"n": 0}

    def flaky(**kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    result = _predict_with_retry(flaky, {}, RateLimiter(None), max_attempts=3)

    assert result == "ok"
    assert attempts["n"] == 3
    # backs off before the 2nd and 3rd attempts, not after the final success
    assert sleeps == [0.5, 1.0]


def test_predict_with_retry_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("earnings_analyser.modules.collapse_step.time.sleep", lambda s: None)

    def always_fails(**kwargs):
        raise RuntimeError("still broken")

    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        _predict_with_retry(always_fails, {}, RateLimiter(None), max_attempts=2)


def test_rate_limiter_disabled_when_max_is_none():
    limiter = RateLimiter(None)
    # should return immediately, no matter how many times it's called
    for _ in range(50):
        limiter.acquire()


def test_rate_limiter_throttles_to_max_per_minute(monkeypatch):
    fake_now = {"t": 0.0}
    monkeypatch.setattr("earnings_analyser.modules.collapse_step.time.monotonic", lambda: fake_now["t"])
    sleeps = []

    def fake_sleep(s):
        sleeps.append(s)
        fake_now["t"] += s

    monkeypatch.setattr("earnings_analyser.modules.collapse_step.time.sleep", fake_sleep)

    limiter = RateLimiter(2)
    limiter.acquire()
    limiter.acquire()
    # a 3rd call within the same 60s window must wait, not proceed immediately
    limiter.acquire()

    assert sleeps  # it actually waited
    assert fake_now["t"] >= 60


def test_collapse_round_uses_one_call_per_group_not_per_dimension(tmp_path):
    """The whole point of DimensionCollapseBundled: a round with N groups
    costs N calls, not N * len(DIMENSIONS)."""
    transcript = _synthetic_transcript(40)
    call_count = {"n": 0}
    lock = threading.Lock()

    def counting_predictor(**kwargs):
        with lock:
            call_count["n"] += 1
        return _fake_predictor(**kwargs)

    with GraphStore(tmp_path / "graph.sqlite") as store:
        collapse(store, counting_predictor, counting_predictor, transcript,
                 window_size=3, window_overlap=1, branching_factor=2, target=2)

    # base level: 1 call per window. Any collapse round after that:
    # 1 call per group, not 1 per (dimension, group) — so total calls
    # must stay well under len(DIMENSIONS)x what a per-dimension version
    # would need. Concretely: base-level windows for 40 sentences,
    # window=3/overlap=1 -> ~20 windows, so >20 calls would indicate the
    # bundling isn't happening.
    assert call_count["n"] < 20 * len(DIMENSIONS)
