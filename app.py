"""Earnings Call Analyzer — Streamlit UI.

Run with: .venv/bin/streamlit run app.py
Requires a local Ollama server running the configured model (see
`src/earnings_analyser/config.py`), or a cloud `dspy.LM` wired up in
`config.configure_dspy()`.

Three phases, tracked in `st.session_state["phase"]`:
  input      — paste/upload a transcript.
  analyzing  — `run_pipeline` runs in a background thread while an
               `st.fragment` polls the (WAL-mode) GraphStore every 1s
               and redraws the graph as levels complete.
  complete   — final report + graph with dimension-tab highlighting.
"""

import hashlib
import tempfile
import threading
import traceback
from pathlib import Path

import streamlit as st

from earnings_analyser.config import configure_dspy
from earnings_analyser.data.pdf_extract import extract_pdf_text
from earnings_analyser.persistence.graph_store import GraphStore
from earnings_analyser.pipeline import run_pipeline
from earnings_analyser.report import build_report
from earnings_analyser.signatures import DIMENSIONS
from earnings_analyser.ui.graph_component import render_graph

# Dimension color lives only inside the graph — one continuous cool arc
# so it reads as a coherent aurora, not a six-color legend. App chrome
# (panels, tabs, buttons) stays navy/violet/paper via .streamlit/config.toml.
DIM_COLORS = {
    "forward_guidance": "#5EEAD4",
    "sentiment": "#818CF8",
    "macro_focus": "#38BDF8",
    "jargon": "#C084FC",
    "uncertainty": "#A78BFA",
    "confidence": "#F472B6",
}
DIMENSION_LABELS = {
    "forward_guidance": "Forward Guidance",
    "uncertainty": "Uncertainty",
    "confidence": "Confidence",
    "sentiment": "Sentiment",
    "macro_focus": "Macro Focus",
    "jargon": "Jargon",
}

st.set_page_config(page_title="Constellation — Earnings Call Analyzer", layout="wide")

# Rounding/backoff and color come from .streamlit/config.toml's native theme
# options; this covers only what theme.toml has no knob for — smooth hover/
# focus transitions and pill-shaped tabs. Kept small and targeted rather than
# a full CSS override, since Streamlit's internal class names aren't a stable
# API across versions.
st.markdown(
    """
    <style>
    @media (prefers-reduced-motion: no-preference) {
      .stButton > button, .stDownloadButton > button {
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124,92,255,0.28); }
      .stButton > button:active { transform: translateY(0); }

      div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
      }
      div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 28px rgba(11,19,48,0.4);
      }

      .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        transition: background 0.2s ease, color 0.2s ease;
      }
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 1.1rem !important; }
    [data-testid="stExpander"] { border-radius: 1rem !important; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [aria-selected="true"] { background: rgba(124,92,255,0.18); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Connecting to the model...")
def get_lm():
    """Returns `(lm, pipeline_profile, backend)` — see `config.configure_dspy`.
    Backend (local Ollama vs. cloud) and its concurrency/batching tuning
    are selected together via `EARNINGS_ANALYSER_BACKEND`; set once at
    process start, not switchable from the UI yet."""
    return configure_dspy()


def _escape_markdown(text: str) -> str:
    return text.replace("$", "\\$")


def _node_payload(node) -> dict:
    return {
        "id": node.node_id,
        "level": node.level,
        "kind": node.kind,
        "dimension": node.dimension,
        "text": node.text,
        "terminal": node.terminal,
    }


def _sync_graph_data(store: GraphStore) -> int | None:
    """Pull anything new since the last poll into session_state. Nodes are
    fetched incrementally per level (once each); edges have no level column
    so they're just refetched in full each tick — cheap at this scale."""
    fetched = st.session_state["levels_fetched"]
    if 0 not in fetched:
        nodes = store.nodes_at_level(0)
        if nodes:
            st.session_state["graph_nodes"].extend(_node_payload(n) for n in nodes)
            fetched.add(0)

    last_level = store.last_complete_level()
    if last_level is not None:
        for level in range(1, last_level + 1):
            if level in fetched:
                continue
            st.session_state["graph_nodes"].extend(_node_payload(n) for n in store.nodes_at_level(level))
            fetched.add(level)

    st.session_state["graph_edges"] = [
        {"child": e.child, "parent": e.parent, "dimension": e.dimension, "weight": e.weight}
        for dim in DIMENSIONS
        for e in store.edges_for_dimension(dim)
    ]
    return last_level


def _start_analysis(transcript_text: str) -> None:
    lm, pipeline_profile, _backend = get_lm()
    digest = hashlib.sha1(transcript_text.encode("utf-8")).hexdigest()[:16]
    db_path = Path(tempfile.gettempdir()) / f"earnings-analyser-{digest}.sqlite"
    for stale in db_path.parent.glob(db_path.name + "*"):
        stale.unlink(missing_ok=True)  # always watch a fresh build, even for a repeat transcript

    status = {"done": False, "error": None, "report": None}

    def _worker() -> None:
        try:
            with GraphStore(db_path) as store:
                run_pipeline(store, transcript_text, lm=lm, **pipeline_profile)
                report = build_report(store, transcript_text)
            status["report"] = report.to_dict()
        except Exception:  # surfaced to the polling fragment via this shared dict
            status["error"] = traceback.format_exc()
        finally:
            status["done"] = True

    threading.Thread(target=_worker, daemon=True).start()

    st.session_state["phase"] = "analyzing"
    st.session_state["db_path"] = db_path
    st.session_state["analysis_status"] = status
    st.session_state["levels_fetched"] = set()
    st.session_state["graph_nodes"] = []
    st.session_state["graph_edges"] = []


def render_input_phase() -> None:
    last_error = st.session_state.pop("last_error", None)
    if last_error:
        st.error("Analysis failed. Details:")
        st.code(last_error, language="text")

    tab_paste, tab_pdf = st.tabs(["Paste transcript", "Upload PDF"])
    transcript_text = None

    with tab_paste:
        pasted = st.text_area("Transcript text", height=280, key="pasted_text")
        if st.button("Analyze", type="primary", key="analyze_paste"):
            if not pasted.strip():
                st.error("Paste transcript text first.")
            else:
                transcript_text = pasted

    with tab_pdf:
        uploaded = st.file_uploader("Transcript PDF", type=["pdf"], key="uploaded_pdf")
        if st.button("Analyze", type="primary", key="analyze_pdf"):
            if uploaded is None:
                st.error("Upload a PDF first.")
            else:
                text = extract_pdf_text(uploaded.read())
                if not text.strip():
                    st.error("Couldn't extract any text from that PDF.")
                else:
                    transcript_text = text

    if transcript_text:
        _start_analysis(transcript_text)
        st.rerun()


@st.fragment(run_every="1s")
def render_analyzing_phase() -> None:
    status = st.session_state["analysis_status"]
    with GraphStore(st.session_state["db_path"]) as store:
        last_level = _sync_graph_data(store)

    level_note = f"level {last_level} composed" if last_level else "reading the transcript..."
    st.caption(f"Building the evidence graph — {level_note}")
    render_graph(st.session_state["graph_nodes"], st.session_state["graph_edges"], active_dim=None, dim_colors=DIM_COLORS)

    if status["done"]:
        if status["error"] is not None:
            st.session_state["last_error"] = status["error"]
            st.session_state["phase"] = "input"
        else:
            st.session_state["report"] = status["report"]
            st.session_state["phase"] = "complete"
            st.session_state["active_dim"] = DIMENSIONS[0]
        st.rerun()


def render_complete_phase() -> None:
    report = st.session_state["report"]

    canvas_col, panel_col = st.columns([1.5, 1], gap="large")

    with panel_col:
        dim = st.radio("Dimension", DIMENSIONS, format_func=lambda d: DIMENSION_LABELS[d], key="active_dim", horizontal=True)
        themes = report["conclusions"][dim]
        if not themes:
            st.caption("No themes surfaced for this dimension.")
        for theme in themes:
            with st.container(border=True):
                st.markdown(f"**{theme['label'] or 'Unlabeled'}**")
                st.markdown(_escape_markdown(theme["narrative"]))
                if theme["evidence"]:
                    with st.expander(f"{len(theme['evidence'])} cited sentence(s)"):
                        for q in sorted(theme["evidence"], key=lambda e: e["weight"], reverse=True):
                            st.markdown(f"**{q['weight']:.2f}** — “{_escape_markdown(q['text'])}”")

    with canvas_col:
        render_graph(st.session_state["graph_nodes"], st.session_state["graph_edges"], active_dim=dim, dim_colors=DIM_COLORS, height=620)

    if st.button("Analyze another transcript"):
        for key in ("phase", "report", "graph_nodes", "graph_edges", "levels_fetched", "active_dim"):
            st.session_state.pop(key, None)
        st.rerun()


def main() -> None:
    _lm, _profile, backend = get_lm()  # warms the cached resource up front

    title_col, backend_col = st.columns([4, 1])
    with title_col:
        st.title("Constellation")
        st.caption("Watch the evidence graph form, dimension by dimension — sentence-grounded, no composite score.")
    with backend_col:
        st.markdown(f"<div style='text-align:right;padding-top:1.2rem;color:#9AA1C4;'>backend: <b>{backend}</b></div>", unsafe_allow_html=True)

    phase = st.session_state.get("phase", "input")
    if phase == "input":
        render_input_phase()
    elif phase == "analyzing":
        render_analyzing_phase()
    else:
        render_complete_phase()


if __name__ == "__main__":
    main()
