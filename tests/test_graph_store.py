from earnings_analyser.persistence.graph_store import Edge, GraphStore, Node


def test_roundtrip_nodes_and_edges(tmp_path):
    db_path = tmp_path / "graph.sqlite"
    with GraphStore(db_path) as store:
        store.add_nodes(
            [
                Node(node_id="s0", level=0, kind="source_sentence", text="Revenue grew.", start=0, end=13),
                Node(node_id="s1", level=0, kind="source_sentence", text="Margins held.", start=14, end=27),
                Node(node_id="c0-uncertainty", level=1, kind="composite", text="No hedging observed.",
                     dimension="uncertainty", label="mild_negative", terminal=True),
            ]
        )
        store.add_edges(
            [
                Edge(child="s0", parent="c0-uncertainty", dimension="uncertainty", weight=0.6),
                Edge(child="s1", parent="c0-uncertainty", dimension="uncertainty", weight=0.4),
            ]
        )
        store.mark_level_complete(1)

        assert store.last_complete_level() == 1

        leaves = store.nodes_at_level(0, kind="source_sentence")
        assert {n.node_id for n in leaves} == {"s0", "s1"}

        terminals = store.terminal_nodes("uncertainty")
        assert len(terminals) == 1
        assert terminals[0].label == "mild_negative"

        fetched = store.get_node("s0")
        assert fetched is not None
        assert fetched.text == "Revenue grew."

        edges = store.edges_for_dimension("uncertainty")
        assert {e.child for e in edges} == {"s0", "s1"}


def test_resumability_reports_none_before_any_level_completes(tmp_path):
    with GraphStore(tmp_path / "graph.sqlite") as store:
        assert store.last_complete_level() is None


def test_wal_mode_is_enabled(tmp_path):
    with GraphStore(tmp_path / "graph.sqlite") as store:
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
