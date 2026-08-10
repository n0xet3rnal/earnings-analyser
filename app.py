"""Earnings Call Analyzer — Streamlit UI.

Run with: .venv/bin/streamlit run app.py
"""

import html as html_lib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import plotly.graph_objects as go
import streamlit as st

from earnings_analyser.config import configure_dspy
from earnings_analyser.data.loader import TranscriptRecord
from earnings_analyser.data.pdf_extract import extract_pdf_text
from earnings_analyser.pipeline import TranscriptScorer
from earnings_analyser.report import build_report
from earnings_analyser.signatures import DIMENSIONS

# Diverging pair (blue=bullish, red=bearish) — polarity of the score.
BLUE = "#2a78d6"
RED = "#e34948"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"

# Categorical, one hue per dimension — identity, not polarity, so it's a
# different color job from the diverging pair above (deliberately skips
# blue/red, slots 1 and 8, to avoid echoing the bullish/bearish colors).
DIM_COLORS = {
    "forward_guidance": "#008300",  # green
    "uncertainty": "#e87ba4",  # magenta
    "confidence": "#eda100",  # yellow
    "sentiment": "#1baf7a",  # aqua
    "macro_focus": "#eb6834",  # orange
    "jargon": "#4a3aa7",  # violet
}

DIMENSION_LABELS = {
    "forward_guidance": "Forward Guidance",
    "uncertainty": "Uncertainty",
    "confidence": "Confidence",
    "sentiment": "Sentiment",
    "macro_focus": "Macro Focus",
    "jargon": "Jargon",
}

st.set_page_config(page_title="Earnings Call Analyzer", layout="wide")


@st.cache_resource(show_spinner="Connecting to Gemini...")
def get_lm():
    return configure_dspy()


@st.cache_data(show_spinner="Scoring transcript with Gemini (this calls the API — may take a minute)...")
def run_analysis(ticker: str, company: str, earnings_date: str, transcript_text: str, n_samples: int) -> dict:
    get_lm()
    # No dataset row backs raw input — eps fields stay None, which is
    # fine: they're only used by the offline BootstrapFewShot calibration
    # metric (against the HF dataset), not by the live scoring pipeline.
    record = TranscriptRecord(
        ticker=ticker or "N/A",
        company=company or ticker or "N/A",
        sector="N/A",
        earnings_date=earnings_date or "N/A",
        year=0.0,
        quarter=0.0,
        eps12mtrailing_eoq=None,
        eps12mfwd_qavg=None,
        transcript=transcript_text,
    )
    scorer = TranscriptScorer(n_samples=n_samples)
    scores = scorer(ticker=record.ticker, earnings_date=record.earnings_date, transcript=record.transcript)
    report = build_report(record, scores)
    return report.to_dict()


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def diverging_bar_chart(section: dict, title: str) -> go.Figure:
    labels = [DIMENSION_LABELS[d] for d in DIMENSIONS]
    values = [section["dimensions"][d]["bullish_score"] for d in DIMENSIONS]
    colors = [BLUE if v >= 0 else RED for v in values]
    hover = [
        f"{DIMENSION_LABELS[d]}<br>Label: {section['dimensions'][d]['label']}<br>"
        f"Score: {section['dimensions'][d]['bullish_score']:+.2f}<br>"
        f"Agreement: {section['dimensions'][d]['agreement']:.0%}"
        for d in DIMENSIONS
    ]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(
            range=[-1.15, 1.15],
            zeroline=True,
            zerolinecolor=AXIS,
            zerolinewidth=1,
            gridcolor=GRID,
            title="Bearish ←  → Bullish",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_SECONDARY, size=11),
        margin=dict(l=10, r=10, t=32, b=32),
        height=260,
        showlegend=False,
    )
    return fig


def render_hero(composite_score: float) -> None:
    if composite_score > 0.05:
        color, label = BLUE, "Bullish"
    elif composite_score < -0.05:
        color, label = RED, "Bearish"
    else:
        color, label = TEXT_MUTED, "Neutral"

    st.markdown(
        f"""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
          <div style="font-size:3rem; font-weight:700; color:{color};">{composite_score:+.2f}</div>
          <div style="font-size:1rem; color:{TEXT_MUTED};">{label} composite score
            <span style="font-size:0.85rem;">(−1 bearish .. +1 bullish, uncertainty upweighted, Q&amp;A weighted higher)</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _escape_markdown(text: str) -> str:
    # Streamlit's markdown renderer treats "$" as a LaTeX delimiter, which
    # mangles dollar figures — exactly the numbers an earnings-call quote
    # can't afford to garble. Escape so quotes/rationale render verbatim.
    return text.replace("$", "\\$")


def render_transcript_pane(transcript_text: str, spans: list[tuple[int, int]], color_hex: str | None) -> None:
    if color_hex is None:
        spans = []
    highlight_bg = _hex_to_rgba(color_hex, 0.35) if color_hex else None
    spans = sorted((s for s in spans if s[0] is not None and s[1] is not None), key=lambda s: s[0])

    parts = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue  # overlapping span from the same dimension — skip, first one wins
        parts.append(html_lib.escape(transcript_text[cursor:start]))
        parts.append(
            f'<mark style="background:{highlight_bg};border-radius:3px;padding:0 1px;">'
            f"{html_lib.escape(transcript_text[start:end])}</mark>"
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


def _toggle_active_dimension(dim: str, section_key: str) -> None:
    # Runs as an on_click callback, BEFORE the script reruns top-to-bottom —
    # unlike `if st.button(...): mutate state`, this guarantees every card
    # in this render pass (including ones drawn earlier in the loop than
    # the one clicked) sees the updated state, not a stale one-run-behind
    # value.
    was_active = (
        st.session_state.get("active_dim") == dim and st.session_state.get("active_section") == section_key
    )
    if was_active:
        st.session_state["active_dim"] = None
        st.session_state["active_section"] = None
    else:
        st.session_state["active_dim"] = dim
        st.session_state["active_section"] = section_key


def render_dimension_cards(section: dict, section_key: str) -> None:
    for dim in DIMENSIONS:
        entry = section["dimensions"][dim]
        is_active = (
            st.session_state.get("active_dim") == dim and st.session_state.get("active_section") == section_key
        )

        with st.container(border=True):
            swatch, label_col, score_col = st.columns([0.06, 0.64, 0.3])
            with swatch:
                st.markdown(
                    f"<div style='width:14px;height:14px;margin-top:8px;border-radius:3px;"
                    f"background:{DIM_COLORS[dim]};'></div>",
                    unsafe_allow_html=True,
                )
            with label_col:
                st.button(
                    f"{DIMENSION_LABELS[dim]} — {entry['label']}",
                    key=f"card_{section_key}_{dim}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                    on_click=_toggle_active_dimension,
                    args=(dim, section_key),
                )
            with score_col:
                st.markdown(
                    f"<div style='text-align:right; padding-top:8px; color:{TEXT_SECONDARY};'>"
                    f"{entry['bullish_score']:+.2f} · agree {entry['agreement']:.0%}</div>",
                    unsafe_allow_html=True,
                )

            if is_active:
                st.markdown(_escape_markdown(entry["rationale"]))
                if entry["evidence"]:
                    st.caption("Evidence — highlighted in the transcript on the right:")
                    for q in entry["evidence"]:
                        icon = "✅" if q["start"] is not None else "⚠️ unverified —"
                        st.markdown(f"{icon} “{_escape_markdown(q['text'])}”")
                else:
                    st.caption("No single quotable passage — the signal is diffuse across the section.")


def main() -> None:
    st.title("Earnings Call Analyzer")
    st.caption(
        "Scores earnings call transcripts across 6 dimensions from academic research "
        "(Matera 2024; Fast Numbers, Slow Language) for stock screening."
    )

    get_lm()

    with st.sidebar:
        st.header("Select a call")
        source = st.radio("Transcript source", ["Paste text", "Upload PDF"])

        with st.expander("Advanced"):
            n_samples = st.slider(
                "Self-consistency samples",
                min_value=1,
                max_value=5,
                value=3,
                help=(
                    "Gemini isn't fully deterministic even at temperature=0, so each section "
                    "is scored this many times and we majority-vote. Higher = more reliable, "
                    "more API calls. 3 is a good default."
                ),
            )

        report_args = None

        if source == "Paste text":
            ticker = st.text_input("Ticker (optional)")
            company = st.text_input("Company name (optional)")
            earnings_date = st.text_input("Earnings date (optional)", placeholder="YYYY-MM-DD")
            transcript_text = st.text_area("Transcript text", height=250)

            if st.button("Analyze", type="primary", use_container_width=True):
                if not transcript_text.strip():
                    st.error("Paste transcript text first.")
                else:
                    report_args = (ticker, company, earnings_date, transcript_text, n_samples)

        else:  # Upload PDF
            ticker = st.text_input("Ticker (optional)")
            company = st.text_input("Company name (optional)")
            earnings_date = st.text_input("Earnings date (optional)", placeholder="YYYY-MM-DD")
            uploaded = st.file_uploader("Transcript PDF", type=["pdf"])

            if st.button("Analyze", type="primary", use_container_width=True):
                if uploaded is None:
                    st.error("Upload a PDF first.")
                else:
                    text = extract_pdf_text(uploaded.read())
                    if not text.strip():
                        st.error(
                            "Couldn't extract any text from that PDF — it may be a scanned "
                            "image without OCR, which this app doesn't handle."
                        )
                    else:
                        report_args = (ticker, company, earnings_date, text, n_samples)

    if report_args is not None:
        st.session_state["report"] = run_analysis(*report_args)
        st.session_state["active_dim"] = None
        st.session_state["active_section"] = None

    report = st.session_state.get("report")
    if report is None:
        st.info("Choose a transcript source in the sidebar, then click Analyze.")
        return

    st.subheader(f"{report['company']} ({report['ticker']}) — {report['earnings_date']}")
    if report["split_confidence"] == "none":
        st.warning(
            "Could not confidently split this transcript into prepared remarks / Q&A — "
            "scores reflect the entire transcript treated as prepared remarks."
        )

    render_hero(report["composite_score"])

    available_sections = [s for s in ["prepared_remarks", "qa"] if report["sections"].get(s)]
    section_labels = {"prepared_remarks": "Prepared Remarks", "qa": "Q&A"}
    if len(available_sections) > 1:
        section_key = st.radio(
            "Analyzing",
            available_sections,
            format_func=lambda s: section_labels[s],
            horizontal=True,
        )
    else:
        section_key = available_sections[0]
        st.caption(f"Analyzing: {section_labels[section_key]} (no Q&A section detected)")

    section = report["sections"][section_key]

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        st.plotly_chart(diverging_bar_chart(section, section_labels[section_key]), use_container_width=True)
        render_dimension_cards(section, section_key)

    with right:
        st.markdown("**Full transcript** — click a dimension on the left to highlight its evidence here.")
        active_dim = st.session_state.get("active_dim")
        active_section = st.session_state.get("active_section")
        spans: list[tuple[int, int]] = []
        color_hex = None
        if active_dim and active_section == section_key:
            evidence = section["dimensions"][active_dim]["evidence"]
            spans = [(q["start"], q["end"]) for q in evidence if q["start"] is not None]
            color_hex = DIM_COLORS[active_dim]
        render_transcript_pane(report["transcript_text"], spans, color_hex)


if __name__ == "__main__":
    main()
