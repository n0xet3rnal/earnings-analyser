"""SQLite-backed, WAL-mode graph store for the collapse pipeline.

WAL mode specifically because a live graph viewer is expected to read the
store while the pipeline is still writing to it — this is exactly the
one-writer/concurrent-readers pattern WAL mode exists for (see
`implementation-plan.md` §5). Resumability comes from `run_status`: a
restarted pipeline reads `last_complete_level()` and skips straight to the
next level instead of recomputing finished work.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    node_id     TEXT PRIMARY KEY,
    level       INTEGER NOT NULL,
    kind        TEXT NOT NULL,
    dimension   TEXT,
    start_off   INTEGER,
    end_off     INTEGER,
    text        TEXT NOT NULL,
    label       TEXT,
    terminal    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS edges (
    child       TEXT NOT NULL REFERENCES nodes(node_id),
    parent      TEXT NOT NULL REFERENCES nodes(node_id),
    dimension   TEXT NOT NULL,
    weight      REAL NOT NULL,
    PRIMARY KEY (child, parent, dimension)
);

CREATE TABLE IF NOT EXISTS run_status (
    level       INTEGER PRIMARY KEY,
    complete    INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nodes_level ON nodes(level);
CREATE INDEX IF NOT EXISTS idx_edges_dimension ON edges(dimension);
CREATE INDEX IF NOT EXISTS idx_edges_parent ON edges(parent);
"""


@dataclass(frozen=True)
class Node:
    node_id: str
    level: int
    kind: str  # "source_sentence" | "composite"
    text: str
    dimension: str | None = None
    start: int | None = None
    end: int | None = None
    label: str | None = None
    terminal: bool = False


@dataclass(frozen=True)
class Edge:
    child: str
    parent: str
    dimension: str
    weight: float


class GraphStore:
    def __init__(self, db_path: str | Path):
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GraphStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- writes -----------------------------------------------------------

    def add_nodes(self, nodes: list[Node]) -> None:
        self._conn.executemany(
            """INSERT OR REPLACE INTO nodes
               (node_id, level, kind, dimension, start_off, end_off, text, label, terminal)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (n.node_id, n.level, n.kind, n.dimension, n.start, n.end, n.text, n.label, int(n.terminal))
                for n in nodes
            ],
        )
        self._conn.commit()

    def add_edges(self, edges: list[Edge]) -> None:
        self._conn.executemany(
            "INSERT OR REPLACE INTO edges (child, parent, dimension, weight) VALUES (?, ?, ?, ?)",
            [(e.child, e.parent, e.dimension, e.weight) for e in edges],
        )
        self._conn.commit()

    def mark_level_complete(self, level: int) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO run_status (level, complete) VALUES (?, 1)", (level,)
        )
        self._conn.commit()

    # -- reads --------------------------------------------------------------

    def last_complete_level(self) -> int | None:
        row = self._conn.execute(
            "SELECT MAX(level) FROM run_status WHERE complete = 1"
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_node(self, node_id: str) -> Node | None:
        row = self._conn.execute(
            "SELECT node_id, level, kind, dimension, start_off, end_off, text, label, terminal FROM nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return Node(node_id=row[0], level=row[1], kind=row[2], dimension=row[3], start=row[4], end=row[5],
                    text=row[6], label=row[7], terminal=bool(row[8]))

    def nodes_at_level(self, level: int, kind: str | None = None) -> list[Node]:
        query = "SELECT node_id, level, kind, dimension, start_off, end_off, text, label, terminal FROM nodes WHERE level = ?"
        params: tuple = (level,)
        if kind is not None:
            query += " AND kind = ?"
            params = (level, kind)
        rows = self._conn.execute(query, params).fetchall()
        return [
            Node(node_id=r[0], level=r[1], kind=r[2], dimension=r[3], start=r[4], end=r[5],
                 text=r[6], label=r[7], terminal=bool(r[8]))
            for r in rows
        ]

    def terminal_nodes(self, dimension: str | None = None) -> list[Node]:
        query = "SELECT node_id, level, kind, dimension, start_off, end_off, text, label, terminal FROM nodes WHERE terminal = 1"
        params: tuple = ()
        if dimension is not None:
            query += " AND dimension = ?"
            params = (dimension,)
        rows = self._conn.execute(query, params).fetchall()
        return [
            Node(node_id=r[0], level=r[1], kind=r[2], dimension=r[3], start=r[4], end=r[5],
                 text=r[6], label=r[7], terminal=bool(r[8]))
            for r in rows
        ]

    def edges_for_dimension(self, dimension: str) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT child, parent, dimension, weight FROM edges WHERE dimension = ?", (dimension,)
        ).fetchall()
        return [Edge(child=r[0], parent=r[1], dimension=r[2], weight=r[3]) for r in rows]
