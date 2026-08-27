"""Deterministic attribution: how much does each source sentence explain
each terminal conclusion (docs/implementation-plan.md §2.4).

Pure code, no model calls. Each level's edge weights are already
normalized (see `modules/collapse_step.py`'s `_normalized_weights`), so a
leaf's total importance to a terminal conclusion is the product of edge
weights along every path from that leaf up to that conclusion, summed
across all paths that reach it — equivalent to chaining each level's
weight matrix together (`W1 @ W2 @ ... @ Wk`), implemented here as a
weighted breadth-first walk down from each terminal node instead of
assembling explicit matrices (same math, no numpy dependency needed at
this node count). No nonlinearity: multiply-then-sum preserves a "share
of the final conclusion this sentence explains" interpretation.
"""

from collections import defaultdict

from ..persistence.graph_store import GraphStore


def compute_attribution(store: GraphStore, dimension: str) -> dict[str, dict[str, float]]:
    """Returns {terminal_node_id: {source_sentence_id: importance}} for
    one dimension — the table the heatmap renders directly."""
    edges = store.edges_for_dimension(dimension)
    children_of: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in edges:
        children_of[e.parent].append((e.child, e.weight))

    result: dict[str, dict[str, float]] = {}
    for terminal in store.terminal_nodes(dimension):
        leaf_importance: dict[str, float] = defaultdict(float)
        frontier: dict[str, float] = {terminal.node_id: 1.0}

        while frontier:
            next_frontier: dict[str, float] = defaultdict(float)
            for node_id, importance in frontier.items():
                children = children_of.get(node_id)
                if not children:
                    # No outgoing edges as a parent means this is a leaf
                    # (source_sentence) — nothing above the leaf layer
                    # reproduces text, so nothing above it can appear here.
                    leaf_importance[node_id] += importance
                    continue
                for child_id, weight in children:
                    next_frontier[child_id] += importance * weight
            frontier = dict(next_frontier)

        result[terminal.node_id] = dict(leaf_importance)

    return result


def compute_node_weights(store: GraphStore, dimension: str) -> dict[str, float]:
    """Returns {sentence_node_id: weight} for every source sentence in
    this dimension's tree — its own edge weight to its immediate
    (base-level) parent, peak-normalized across the dimension.

    Deliberately just the one hop, not a multi-level backprop: chaining
    weights down from a terminal (as an earlier version of this function
    did) makes a sentence's score shrink with every hop and sibling its
    path happens to pass through — a tree-shape artifact, not a
    relevance signal. This is the model's direct, single-hop relevance
    judgment for that sentence, which is what a relevance filter should
    actually threshold on. Composites/terminals get no score of their
    own here; the UI derives their visibility bottom-up instead — a
    composite stays visible if any sentence beneath it passes the
    threshold (see `graph_frontend/index.html`'s `recomputeAliveSet`).
    """
    sentence_ids = {n.node_id for n in store.nodes_at_level(0)}
    raw: dict[str, float] = defaultdict(float)
    for e in store.edges_for_dimension(dimension):
        if e.child in sentence_ids:
            raw[e.child] = max(raw[e.child], e.weight)

    peak = max(raw.values(), default=1.0) or 1.0
    return {node_id: score / peak for node_id, score in raw.items()}
