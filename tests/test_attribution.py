import pytest

from earnings_analyser.analysis.attribution import compute_attribution
from earnings_analyser.persistence.graph_store import Edge, GraphStore, Node


def test_single_level_attribution_matches_edge_weights(tmp_path):
    with GraphStore(tmp_path / "graph.sqlite") as store:
        store.add_nodes(
            [
                Node(node_id="s0", level=0, kind="source_sentence", text="a", start=0, end=1),
                Node(node_id="s1", level=0, kind="source_sentence", text="b", start=1, end=2),
                Node(node_id="t0", level=1, kind="composite", text="summary", dimension="uncertainty",
                     label="neutral", terminal=True),
            ]
        )
        store.add_edges(
            [
                Edge(child="s0", parent="t0", dimension="uncertainty", weight=0.6),
                Edge(child="s1", parent="t0", dimension="uncertainty", weight=0.4),
            ]
        )

        attribution = compute_attribution(store, "uncertainty")

        assert attribution == {"t0": {"s0": pytest.approx(0.6), "s1": pytest.approx(0.4)}}


def test_two_level_chain_multiplies_and_sums_paths(tmp_path):
    # s0,s1 -> c0 (level 1)   s2,s3 -> c1 (level 1)   c0,c1 -> t0 (level 2, terminal)
    with GraphStore(tmp_path / "graph.sqlite") as store:
        store.add_nodes(
            [
                Node(node_id=f"s{i}", level=0, kind="source_sentence", text=str(i), start=i, end=i + 1)
                for i in range(4)
            ]
            + [
                Node(node_id="c0", level=1, kind="composite", text="c0", dimension="uncertainty"),
                Node(node_id="c1", level=1, kind="composite", text="c1", dimension="uncertainty"),
                Node(node_id="t0", level=2, kind="composite", text="t0", dimension="uncertainty",
                     label="neutral", terminal=True),
            ]
        )
        store.add_edges(
            [
                Edge(child="s0", parent="c0", dimension="uncertainty", weight=0.7),
                Edge(child="s1", parent="c0", dimension="uncertainty", weight=0.3),
                Edge(child="s2", parent="c1", dimension="uncertainty", weight=0.5),
                Edge(child="s3", parent="c1", dimension="uncertainty", weight=0.5),
                Edge(child="c0", parent="t0", dimension="uncertainty", weight=0.6),
                Edge(child="c1", parent="t0", dimension="uncertainty", weight=0.4),
            ]
        )

        attribution = compute_attribution(store, "uncertainty")["t0"]

        assert attribution["s0"] == pytest.approx(0.7 * 0.6)
        assert attribution["s1"] == pytest.approx(0.3 * 0.6)
        assert attribution["s2"] == pytest.approx(0.5 * 0.4)
        assert attribution["s3"] == pytest.approx(0.5 * 0.4)
        assert sum(attribution.values()) == pytest.approx(1.0)


def test_multi_parent_convergence_sums_across_paths(tmp_path):
    # s0 is shared (window-boundary overlap) between c0 and c1, both feeding t0.
    with GraphStore(tmp_path / "graph.sqlite") as store:
        store.add_nodes(
            [
                Node(node_id="s0", level=0, kind="source_sentence", text="shared", start=0, end=1),
                Node(node_id="s1", level=0, kind="source_sentence", text="only-c0", start=1, end=2),
                Node(node_id="s2", level=0, kind="source_sentence", text="only-c1", start=2, end=3),
                Node(node_id="c0", level=1, kind="composite", text="c0", dimension="jargon"),
                Node(node_id="c1", level=1, kind="composite", text="c1", dimension="jargon"),
                Node(node_id="t0", level=2, kind="composite", text="t0", dimension="jargon",
                     label="neutral", terminal=True),
            ]
        )
        store.add_edges(
            [
                Edge(child="s0", parent="c0", dimension="jargon", weight=0.5),
                Edge(child="s1", parent="c0", dimension="jargon", weight=0.5),
                Edge(child="s0", parent="c1", dimension="jargon", weight=0.4),
                Edge(child="s2", parent="c1", dimension="jargon", weight=0.6),
                Edge(child="c0", parent="t0", dimension="jargon", weight=0.5),
                Edge(child="c1", parent="t0", dimension="jargon", weight=0.5),
            ]
        )

        attribution = compute_attribution(store, "jargon")["t0"]

        expected_s0 = (0.5 * 0.5) + (0.4 * 0.5)  # sum across both converging paths
        assert attribution["s0"] == pytest.approx(expected_s0)
