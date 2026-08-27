"""D3-force graph visualization, as a real (bidirectional) Streamlit
component — `graph_frontend/index.html`, plain JS, no build tooling.

This is deliberately not `st.components.v1.html`: that API re-embeds a
fresh iframe from scratch on every call, so a dimension switch or a new
poll tick meant tearing down and rebuilding the whole simulation and DOM
— visible as a full-graph flash/jitter, no drag, and no way to add
streamed-in nodes without redoing everyone else's layout too.
`declare_component` keeps the same iframe alive across reruns from the
same call site and just delivers new args to it via postMessage, so the
frontend can diff nodes/edges itself, merge in only what's new, and treat
a dimension switch as a pure restyle. See `graph_frontend/index.html` for
the actual protocol/merge/highlight logic.
"""

import os

import streamlit.components.v1 as components

_SOURCE_COLOR = "#6E6684"  # neutral, desaturated violet — reads as part of the purple-on-black family, not an outlier

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graph_frontend")
_component = components.declare_component("earnings_graph", path=_COMPONENT_DIR)


def render_graph(
    nodes: list[dict],
    edges: list[dict],
    active_dim: str | None,
    dim_colors: dict[str, str],
    height: int = 480,
) -> dict | None:
    """Returns the last value the frontend sent via `Streamlit.setComponentValue`
    — currently just a clicked source-sentence node, `{sentenceId, start, end}`
    (see the `click` handler in `graph_frontend/index.html`), or `None` if
    nothing's been clicked yet this session."""
    return _component(
        nodes=nodes,
        edges=edges,
        activeDim=active_dim,
        dimColors=dim_colors,
        sourceColor=_SOURCE_COLOR,
        height=height,
        key="earnings_graph",
        default=None,
    )
