"""Earnings Call Analyzer — Streamlit UI.

Run with: .venv/bin/streamlit run app.py
Requires a local Ollama server running the configured model (see
`src/earnings_analyser/config.py`).
"""

import hashlib
import html as html_lib
import tempfile
from pathlib import Path

import streamlit as st

from earnings_analyser.config import configure_dspy
from earnings_analyser.data.pdf_extract import extract_pdf_text
from earnings_analyser.persistence.graph_store import GraphStore
from earnings_analyser.pipeline import run_pipeline
from earnings_analyser.report import build_report
from earnings_analyser.signatures import DIMENSIONS

DIM_COLORS = {
    "forward_guidance": "#008300",
    "uncertainty": "#e87ba4",
    "confidence": "#eda100",
    "sentiment": "#1baf7a",
    "macro_focus": "#eb6834",
    "jargon": "#4a3aa7",
}
DIMENSION_LABELS = {
    "forward_guidance": "Forward Guidance",
    "uncertainty": "Uncertainty",
    "confidence": "Confidence",
    "sentiment": "Sentiment",
    "macro_focus": "Macro Focus",
    "jargon": "Jargon",
}
LABEL_COLORS = {
    "strong_negative": "#e34948",
    "mild_negative": "#eda100",
    "neutral": "#898781",
    "mild_positive": "#5ea3e0",
    "strong_positive": "#2a78d6",
}
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"

st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")


@st.cache_resource(show_spinner="Connecting to local model...")
def get_lm():
    return configure_dspy()


def run_analysis(ticker: str, company: str, earnings_date: str, transcript_text: str) -> dict:
    get_lm()
    digest = hashlib.sha1(transcript_text.encode("utf-8")).hexdigest()[:16]
    db_path = Path(tempfile.gettempdir()) / f"earnings-analyser-{digest}.sqlite"

    with GraphStore(db_path) as store:
        run_pipeline(store, transcript_text)
        report = build_report(store, transcript_text, ticker or "N/A", company or ticker or "N/A", earnings_date or "N/A")
    return report.to_dict()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _escape_markdown(text: str) -> str:
    return text.replace("$", "\\$")


def render_heatmap_pane(transcript_text: str, evidence: list[dict], color_hex: str) -> None:
    spans = sorted(
        ((e["start"], e["end"], e["weight"]) for e in evidence if e["start"] is not None),
        key=lambda s: s[0],
    )
    max_weight = max((w for _, _, w in spans), default=1.0) or 1.0

    parts = []
    cursor = 0
    for start, end, weight in spans:
        if start < cursor:
            continue
        parts.append(html_lib.escape(transcript_text[cursor:start]))
        alpha = 0.15 + 0.65 * (weight / max_weight)
        bg = _hex_to_rgba(color_hex, alpha)
        parts.append(
            f'<mark style="background:{bg};border-radius:3px;padding:0 1px;" '
            f'title="importance: {weight:.3f}">{html_lib.escape(transcript_text[start:end])}</mark>'
        )
        cursor = end
    parts.append(html_lib.escape(transcript_text[cursor:]))

    body = "".join(parts).replace("\n", "<br>")
    scroll_js = (
        "<script>const m=document.querySelector('mark'); "
        "if(m){m.scrollIntoView({block:'center',behavior:'smooth'});}</script>"
        if spans
        else ""
    )
    html = f"""
    <div style="font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; font-size:14px;
                line-height:1.7; color:#0b0b0b; padding: 4px 10px; background:#fcfcfb;">
      {body}
    </div>
    {scroll_js}
    """
    st.components.v1.html(html, height=720, scrolling=True)


def _toggle_active(dim: str, theme_index: int) -> None:
    was_active = st.session_state.get("active_dim") == dim and st.session_state.get("active_theme") == theme_index
    if was_active:
        st.session_state["active_dim"] = None
        st.session_state["active_theme"] = None
    else:
        st.session_state["active_dim"] = dim
        st.session_state["active_theme"] = theme_index


def render_conclusion_cards(dim: str, themes: list[dict]) -> None:
    for i, theme in enumerate(themes):
        is_active = st.session_state.get("active_dim") == dim and st.session_state.get("active_theme") == i
        label = theme["label"]

        with st.container(border=True):
            swatch, label_col = st.columns([0.06, 0.94])
            with swatch:
                chip_color = LABEL_COLORS.get(label, TEXT_MUTED)
                st.markdown(
                    f"<div style='width:14px;height:14px;margin-top:8px;border-radius:3px;"
                    f"background:{chip_color};'></div>",
                    unsafe_allow_html=True,
                )
            with label_col:
                st.button(
                    f"Theme {i + 1} — {label or 'unlabeled'}",
                    key=f"card_{dim}_{i}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    on_click=_toggle_active,
                    args=(dim, i),
                )
            if is_active:
                st.markdown(_escape_markdown(theme["narrative"]))
                if theme["evidence"]:
                    st.caption("Evidence — highlighted by importance in the transcript on the right:")
                    for q in sorted(theme["evidence"], key=lambda e: e["weight"], reverse=True):
                        st.markdown(f"**{q['weight']:.2f}** — “{_escape_markdown(q['text'])}”")
                else:
                    st.caption("No cited evidence for this theme.")


def main() -> None:
    st.title("Earnings Call Analyzer")
    st.caption("Sentence-grounded, cited analysis across 6 dimensions — recursively collapsed evidence, no composite score.")

    get_lm()

    with st.sidebar:
        st.header("Select a call")
        source = st.radio("Transcript source", ["Paste text", "Upload PDF"])
        ticker = st.text_input("Ticker (optional)")
        company = st.text_input("Company name (optional)")
        earnings_date = st.text_input("Earnings date (optional)", placeholder="YYYY-MM-DD")

        report_args = None

        if source == "Paste text":
            transcript_text = st.text_area("Transcript text", height=250)
            if st.button("Analyze", type="primary", use_container_width=True):
                if not transcript_text.strip():
                    st.error("Paste transcript text first.")
                else:
                    report_args = (ticker, company, earnings_date, transcript_text)
        else:
            uploaded = st.file_uploader("Transcript PDF", type=["pdf"])
            if st.button("Analyze", type="primary", use_container_width=True):
                if uploaded is None:
                    st.error("Upload a PDF first.")
                else:
                    text = extract_pdf_text(uploaded.read())
                    if not text.strip():
                        st.error("Couldn't extract any text from that PDF.")
                    else:
                        report_args = (ticker, company, earnings_date, text)

    if report_args is not None:
        with st.spinner("Collapsing transcript into grounded conclusions (this runs the local model — may take a while)..."):
            st.session_state["report"] = run_analysis(*report_args)
        st.session_state["active_dim"] = None
        st.session_state["active_theme"] = None

    report = st.session_state.get("report")
    if report is None:
        st.info("Choose a transcript source in the sidebar, then click Analyze.")
        return

    st.subheader(f"{report['company']} ({report['ticker']}) — {report['earnings_date']}")

    dim = st.radio("Dimension", DIMENSIONS, format_func=lambda d: DIMENSION_LABELS[d], horizontal=True)
    themes = report["conclusions"][dim]

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.markdown(f"**{DIMENSION_LABELS[dim]}** — {len(themes)} theme(s)")
        render_conclusion_cards(dim, themes)

    with right:
        st.markdown("**Full transcript** — click a theme on the left to see its evidence heatmap here.")
        active_dim = st.session_state.get("active_dim")
        active_theme = st.session_state.get("active_theme")
        if active_dim == dim and active_theme is not None:
            evidence = themes[active_theme]["evidence"]
            render_heatmap_pane(report["transcript_text"], evidence, DIM_COLORS[dim])
        else:
            render_heatmap_pane(report["transcript_text"], [], DIM_COLORS[dim])


if __name__ == "__main__":
    main()
