"""CallSense: Streamlit UI.

Run with: .venv/bin/streamlit run app.py
Requires a local Ollama server running the configured model (see
`src/earnings_analyser/config.py`), or a cloud `dspy.LM` wired up in
`config.configure_dspy()`.

Two top-level phases, tracked in `st.session_state["phase"]`:
  input — paste/upload a transcript.
  (anything else) — `render_graph_phase`, one `st.fragment` that covers
      both "analyzing" and "complete" internally via `status["done"]`,
      never crossing a full `st.rerun()` between them. `run_pipeline` runs
      in a background thread while the fragment polls the (WAL-mode)
      GraphStore every 1s and shows a progress bar plus a growing log of
      completed levels (no live graph — it was too fragile to render
      mid-build); once done, the same fragment mounts the graph for the
      first time and renders the report panel and dimension-tab
      highlighting — kept in one fragment simply to avoid a full-page
      rerun on every 1s poll tick (see `render_graph_phase`).
"""

import hashlib
import html
import re
import tempfile
import threading
import time
import traceback
import urllib.parse
from pathlib import Path

import streamlit as st

from earnings_analyser.config import BACKEND_PROFILES, configure_dspy
from earnings_analyser.data.pdf_extract import extract_pdf_text
from earnings_analyser.modules.collapse_step import estimate_call_count
from earnings_analyser.persistence.graph_store import GraphStore
from earnings_analyser.pipeline import run_pipeline
from earnings_analyser.report import build_report
from earnings_analyser.signatures import DIMENSIONS
from earnings_analyser.ui.graph_component import render_graph

# Dimension color lives only inside the graph — one continuous cool arc
# so it reads as a coherent aurora, not a six-color legend. App chrome
# (panels, tabs, buttons) stays purple-on-black via .streamlit/config.toml.
# Kept within a jewel-toned, cool family close to the app's own violet
# accent (#8B5CF6) rather than a wide rainbow — no warm oranges/yellows.
DIM_COLORS = {
    "forward_guidance": "#2DD4BF",
    "sentiment": "#8B7CF6",
    "macro_focus": "#5B8DEF",
    "jargon": "#B565F0",
    "uncertainty": "#9D7BEB",
    "confidence": "#D6669B",
}
DIMENSION_LABELS = {
    "forward_guidance": "Forward Guidance",
    "uncertainty": "Uncertainty",
    "confidence": "Confidence",
    "sentiment": "Sentiment",
    "macro_focus": "Macro Focus",
    "jargon": "Jargon",
}

st.set_page_config(page_title="CallSense", layout="wide")

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
      .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(139,92,246,0.28); }
      .stButton > button:active { transform: translateY(0); }

      div[data-testid="stVerticalBlockBorderWrapper"] {
        transition: box-shadow 0.2s ease, border-color 0.2s ease;
      }
      div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 8px 28px rgba(0,0,0,0.5);
      }

      .stTabs [data-testid="stTab"] {
        transition: background 0.2s ease, color 0.2s ease;
      }
    }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 1.1rem !important; }
    [data-testid="stExpander"] { border-radius: 1rem !important; overflow: hidden; }
    .stTabs [role="tablist"] { gap: 4px; }
    .stTabs [data-testid="stTab"] {
      border-radius: 999px !important;
      padding: 6px 18px !important;
      height: auto !important;
    }
    .stTabs [data-testid="stTab"][aria-selected="true"] { background: rgba(139,92,246,0.18); }

    /* Dimension picker: a plain st.radio, restyled to read as a vertical
       tab list (left accent bar, no radio dot) rather than a form control. */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
      flex-direction: column;
      gap: 2px;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"] {
      padding: 10px 12px;
      border-radius: 0.6rem;
      border-left: 3px solid transparent;
      line-height: 1.25;
      transition: background 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"]:hover {
      background: rgba(139,92,246,0.08);
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"][data-selected="true"] {
      background: rgba(139,92,246,0.18);
      border-left-color: #8B5CF6;
    }
    div[data-testid="stRadio"] label[data-testid="stRadioOption"] > div > div > div:first-child {
      display: none;
    }
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
        "start": node.start,
        "end": node.end,
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


def _replay_cached_run(cache_store: GraphStore, dest_store: GraphStore, status: dict, pace: float = 0.6) -> None:
    """Re-plays a previously-completed run into a fresh store, level by
    level with a pause between — same shape the live pipeline writes in,
    so the polling fragment streams it exactly like a real run. Lets UI
    changes get exercised repeatedly without spending LLM calls/rate limit.

    No real LLM calls happen here, but `status["calls_done"]` still
    advances (one "call" per `len(DIMENSIONS)` composites written at a
    level — matches how `collapse_step._write_dimension_result` actually
    writes one composite per dimension per call) so the progress bar
    stays meaningful while testing against the fixture instead of just
    sitting dark for that path. Single-threaded (unlike the real
    pipeline's worker pool), so no lock needed for this increment."""
    seen_ids: set[str] = set()
    all_edges = [e for dim in DIMENSIONS for e in cache_store.edges_for_dimension(dim)]

    nodes0 = cache_store.nodes_at_level(0)
    dest_store.add_nodes(nodes0)
    seen_ids.update(n.node_id for n in nodes0)

    last_level = cache_store.last_complete_level() or 0
    for level in range(1, last_level + 1):
        time.sleep(pace)
        nodes = cache_store.nodes_at_level(level)
        dest_store.add_nodes(nodes)
        seen_ids.update(n.node_id for n in nodes)
        dest_store.add_edges([e for e in all_edges if e.child in seen_ids and e.parent in seen_ids])
        dest_store.mark_level_complete(level)
        status["calls_done"] += len(nodes) // len(DIMENSIONS)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_run.sqlite"
# scripts/generate_ui_fixture.py builds the fixture DB from this exact
# file — every node's start/end offset in it is relative to this text, not
# to a placeholder string, so the transcript-jump feature needs the real
# thing here too.
FIXTURE_TRANSCRIPT_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_transcript.txt"


def _estimated_calls(transcript_text: str, pipeline_profile: dict) -> int:
    return estimate_call_count(
        transcript_text,
        window_size=pipeline_profile["window_size"],
        window_overlap=pipeline_profile["window_overlap"],
        branching_factor=pipeline_profile["branching_factor"],
        target=pipeline_profile["target"],
    )


def _start_analysis(transcript_text: str = "", use_fixture: bool = False) -> None:
    digest = hashlib.sha1((transcript_text or "fixture").encode("utf-8")).hexdigest()[:16]
    db_path = Path(tempfile.gettempdir()) / f"earnings-analyser-{digest}.sqlite"
    for stale in db_path.parent.glob(db_path.name + "*"):
        stale.unlink(missing_ok=True)  # always watch a fresh build, even for a repeat transcript

    # Computed up front (cheap — pure arithmetic, see estimate_call_count)
    # so the progress bar has a real denominator from the very first poll
    # tick, not just once the first level lands. The fixture path was
    # always generated against the cloud profile (scripts/generate_ui_fixture.py),
    # regardless of whatever backend is currently configured, so its
    # estimate uses that profile specifically rather than get_lm()'s.
    if use_fixture:
        calls_total = _estimated_calls(FIXTURE_TRANSCRIPT_PATH.read_text(), BACKEND_PROFILES["cloud"]["pipeline"])
    else:
        _lm, pipeline_profile, _backend = get_lm()
        calls_total = _estimated_calls(transcript_text, pipeline_profile)

    status = {"done": False, "error": None, "report": None, "calls_done": 0, "calls_total": calls_total}
    progress_lock = threading.Lock()  # ThreadPoolExecutor workers call this concurrently; += isn't atomic on its own

    def _on_progress() -> None:
        with progress_lock:
            status["calls_done"] += 1

    def _worker() -> None:
        nonlocal transcript_text
        try:
            with GraphStore(db_path) as store:
                if use_fixture:
                    with GraphStore(FIXTURE_PATH) as fixture_store:
                        _replay_cached_run(fixture_store, store, status)
                    transcript_text = FIXTURE_TRANSCRIPT_PATH.read_text()
                else:
                    lm, pipeline_profile, _backend = get_lm()
                    run_pipeline(store, transcript_text, lm=lm, on_progress=_on_progress, **pipeline_profile)
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
    st.session_state["analysis_log"] = ["Reading transcript..."]
    st.session_state["logged_level"] = -1


def render_input_phase() -> None:
    # Centered, not stretched edge-to-edge — a form this short doesn't
    # need the full width of a wide layout.
    _left, center, _right = st.columns([1, 2, 1])
    with center:
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

        st.divider()
        if st.button("Test with sample data (no LLM call)", key="analyze_fixture"):
            _start_analysis(use_fixture=True)
            st.rerun()


GRAPH_HEIGHT = 620


def _render_transcript_panel(transcript_text: str, selected: dict | None) -> None:
    """Always visible, below the graph — a placeholder until a source-
    sentence node is clicked (see the `click` handler in
    `ui/graph_frontend/index.html`), then the full transcript with that
    sentence highlighted and scrolled into view. A one-shot `st.iframe`
    render (via a data: URI — `st.components.v1.html` is past its removal
    date in this Streamlit version and was silently producing no iframe
    at all), not the persistent declared component the graph uses — this
    doesn't need incremental state, and re-rendering a short HTML blob on
    each click is expected, not the "no redraw" case that mattered for
    the graph/weight-slider."""
    st.divider()
    st.caption("Transcript")
    if not selected:
        st.caption("Click a sentence node in the graph to jump to it here.")
        return

    # PDF-extracted transcripts carry a hard newline roughly every ~80
    # characters — the source PDF's own line wrapping, unrelated to
    # sentence boundaries. Rendered verbatim under `white-space: pre-wrap`
    # (needed to keep genuine paragraph breaks), every one of those forces
    # an early line break, which is why the panel read as using only
    # half its width. Swapping lone newlines for spaces is a 1-for-1
    # character substitution, so it doesn't shift any sentence's
    # start/end offset — genuine paragraph breaks (blank lines) are left
    # alone.
    display_text = re.sub(r"(?<!\n)\n(?!\n)", " ", transcript_text)

    start, end = selected["start"], selected["end"]
    before = html.escape(display_text[:start])
    target = html.escape(display_text[start:end])
    after = html.escape(display_text[end:])
    doc = f"""<!doctype html><html><body style="margin:0;background:transparent;">
    <div id="transcript-scroll" style="max-height:320px;width:100%;overflow-y:auto;padding:16px 20px;
         background:#0F0D16;border:1px solid #211C2E;border-radius:0.8rem;
         font-family:'Libre Baskerville',Georgia,serif;color:#B9B2CC;line-height:1.7;
         white-space:pre-wrap;box-sizing:border-box;">{before}<mark id="jump-target"
         style="background:#8B5CF6;color:#06060A;padding:0 2px;border-radius:3px;">{target}</mark>{after}</div>
    <script>document.getElementById('jump-target').scrollIntoView({{block: 'center'}});</script>
    </body></html>
    """
    st.iframe(src="data:text/html;charset=utf-8," + urllib.parse.quote(doc), height=340)


# Analyzing and complete are ONE fragment, not two functions crossed by a
# st.rerun() — simply to avoid a full-page rerun on every 1s poll tick.
# The graph itself is never mounted until analysis completes, so there's
# no remount concern to design around here.
@st.fragment(run_every="1s")
def render_graph_phase() -> None:
    status = st.session_state["analysis_status"]
    # Always sync, even after status["done"] flips true — the worker sets
    # that flag right after its own GraphStore closes, within microseconds
    # of the final mark_level_complete/add_edges commit, so the poll tick
    # that first observes done=True can otherwise be the one tick that
    # never reads the final level's data at all (a real bug: it meant the
    # terminal composites/hub edges for the last level sometimes never
    # made it into graph_nodes/graph_edges before rendering stopped syncing).
    with GraphStore(st.session_state["db_path"]) as store:
        last_level = _sync_graph_data(store)
        if last_level is not None and last_level > st.session_state["logged_level"]:
            st.session_state["logged_level"] = last_level
            st.session_state["analysis_log"].append(f"Level {last_level} composed")
        if status["done"] and status["error"] is None and "report" not in st.session_state:
            st.session_state["analysis_log"].append("Finalizing report...")
            st.session_state["report"] = status["report"]
            st.session_state["active_dim"] = DIMENSIONS[0]
            st.session_state["analysis_log"].append("Analysis complete.")

    if status["done"] and status["error"] is not None:
        st.session_state["last_error"] = status["error"]
        st.session_state["phase"] = "input"
        st.rerun()
        return

    if not status["done"]:
        _left, mid, _right = st.columns([1, 2, 1])
        with mid:
            with st.status("Analyzing transcript...", expanded=True, state="running"):
                calls_total = max(status["calls_total"], 1)
                calls_done = min(status["calls_done"], calls_total)  # clamp defensively against any estimate/actual drift
                st.progress(calls_done / calls_total, text=f"{calls_done} / {calls_total} calls")
                for line in st.session_state["analysis_log"]:
                    st.write(line)
        return

    canvas_col, panel_col = st.columns([1.5, 1], gap="large")
    report = st.session_state["report"]
    with panel_col:
        tab_col, cards_col = st.columns([2, 3], gap="medium")
        with tab_col:
            dim = st.radio("Dimension", DIMENSIONS, format_func=lambda d: DIMENSION_LABELS[d],
                            key="active_dim", horizontal=False, label_visibility="collapsed")
        with cards_col:
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
        clicked = render_graph(st.session_state["graph_nodes"], st.session_state["graph_edges"], active_dim=dim, dim_colors=DIM_COLORS, height=GRAPH_HEIGHT)
        if clicked and clicked != st.session_state.get("selected_sentence"):
            st.session_state["selected_sentence"] = clicked

    _render_transcript_panel(report["transcript_text"], st.session_state.get("selected_sentence"))

    st.download_button(
        "Save this run",
        data=st.session_state["db_path"].read_bytes(),
        file_name=st.session_state["db_path"].name,
        mime="application/octet-stream",
    )

    if st.button("Analyze another transcript"):
        for key in ("phase", "report", "graph_nodes", "graph_edges", "levels_fetched", "active_dim",
                    "analysis_status", "db_path", "selected_sentence", "analysis_log", "logged_level"):
            st.session_state.pop(key, None)
        st.session_state["phase"] = "input"
        st.rerun()


def main() -> None:
    _lm, _profile, backend = get_lm()  # warms the cached resource up front

    title_col, backend_col = st.columns([4, 1])
    with title_col:
        st.title("CallSense")
        st.caption("Explore the evidence graph, dimension by dimension: sentence-grounded, no composite score.")
    with backend_col:
        st.markdown(f"<div style='text-align:right;padding-top:1.2rem;color:#9C93B5;'>backend: <b>{backend}</b></div>", unsafe_allow_html=True)

    phase = st.session_state.get("phase", "input")
    if phase == "input":
        render_input_phase()
    else:
        render_graph_phase()


if __name__ == "__main__":
    main()
