# Earnings Call Analyzer

Scores earnings call transcripts across six linguistic dimensions drawn from academic research on analyst communication, for use as a stock screening signal. Built with DSPy against Google Gemini.

## Academic basis

**Matera 2024** ([arXiv:2511.15214](https://arxiv.org/abs/2511.15214)) defines the six scoring dimensions used here: Forward Guidance, Uncertainty, Confidence, Sentiment, Macro Focus, and Jargon. Its central finding is that analysts systematically underweight uncertainty language when interpreting a call. This project follows that finding directly: the composite score weights Uncertainty twice as heavily as most other dimensions.

**"Fast Numbers, Slow Language"** ([arXiv:2606.29734](https://arxiv.org/abs/2606.29734)) provides the scoring scale used for each dimension (a bounded range from strongly negative to strongly positive, discretized here into five anchor levels) and establishes that the Q&A portion of a call carries more signal than the prepared remarks. The composite score reflects this too: Q&A is weighted 60/40 over prepared remarks.

## What it does

Given a transcript (pasted text or an uploaded PDF), the pipeline:

1. Splits the transcript into prepared remarks and Q&A.
2. Scores each section independently across all six dimensions.
3. Combines the twelve resulting scores into a single composite, upweighting uncertainty and the Q&A section per the papers above.
4. Surfaces the evidence behind every score: verbatim quotes located and verified in the original transcript, not paraphrased summaries.

The output is a Streamlit app showing the composite score, a per-dimension breakdown chart, and the full transcript with the evidence for whichever dimension you're inspecting highlighted in place.

## Approach

### Discrete scoring instead of raw floats

Each dimension is scored by asking Gemini to classify a section into one of five discrete levels (`strong_negative`, `mild_negative`, `neutral`, `mild_positive`, `strong_positive`), which map to fixed values (-1.0, -0.3, 0.0, 0.3, 1.0). This was a deliberate choice over asking the model to output a raw number directly. LLMs are reasonably reliable classifiers but not reliable at inventing calibrated continuous numbers on the fly, and a fixed five-point scale matches how the source papers actually define their anchors.

### Self-consistency for reliability

Gemini's output is not fully deterministic even at `temperature=0`. This was confirmed empirically during development: identical inputs occasionally produced different labels on repeated calls. To compensate, each section is scored multiple times (3 by default, configurable) at a moderate temperature with caching disabled, and the majority label is taken. The fraction of samples agreeing with the majority is recorded as an `agreement` score, which doubles as a rough confidence signal in the UI.

### Verified evidence, not trusted quotes

Each dimension score is accompanied by short verbatim quotes the model claims support its judgment. These are not taken on trust. Every quote is checked against the actual source text (tolerant of whitespace and smart-quote differences, not tolerant of altered wording), and only verified quotes are highlighted in the transcript view. Quotes that fail verification are shown but flagged, since this usually means the model paraphrased rather than quoted. Evidence is pooled across all self-consistency samples that agreed with the majority label rather than taken from a single sample, since different samples often surface different valid supporting passages even when they reach the same conclusion.

### Two-stage transcript splitting

Splitting a transcript into prepared remarks and Q&A is not done with fixed regex patterns as the primary method, since transcript formatting varies by source (different vendors, company-published PDFs, OCR output). Instead, a DSPy signature reads the transcript and returns a short verbatim marker for where Q&A begins; that marker is then located in the source text using the same verification approach as evidence quotes, and the split happens at that exact, confirmed offset. A regex-based splitter (tuned against the demo dataset's phrasing conventions) exists only as a fallback if the model call fails outright.

### Composite scoring

Each dimension is scored in its own natural direction (for example, Uncertainty scores higher when there is more hedging, not less). A per-dimension sign flag determines whether a high raw score is bullish or bearish before it enters the weighted composite, keeping each dimension's own meaning intuitive while still producing a single directional score.

## Tech stack

- **DSPy** for structured LLM signatures and (eventually) prompt optimization
- **Google Gemini** (`gemini-flash-lite-latest`) as the underlying model, accessed through DSPy's LM wrapper (litellm)
- **Streamlit** for the UI, with Plotly for the dimension breakdown charts
- **HuggingFace `datasets`**, specifically `glopardo/sp500-earnings-transcripts`, as a calibration corpus (not a live data source; see Limitations)
- **pypdf** for PDF text extraction
- **uv** for dependency management, `requirements.txt` for the tracked dependency list

## Project structure

```
app.py                          Streamlit UI
src/earnings_analyser/
  config.py                     DSPy + Gemini configuration
  pipeline.py                   End-to-end orchestration: split -> score -> aggregate
  composite.py                  Weighted aggregation of per-dimension, per-section scores
  report.py                     Serializable report structure consumed by the UI
  text_matching.py              Verified substring location, used for evidence and section splitting
  data/
    loader.py                   HuggingFace dataset loading (calibration use only)
    pdf_extract.py               PDF to text extraction
    splitter.py                  Regex-based transcript splitter (fallback only)
  modules/
    section_scorer.py           Runs the scoring signature with self-consistency voting
    section_splitter.py         LLM-based transcript splitter (primary)
  signatures/
    dimension_scores.py         The combined 6-dimension DSPy signature
    section_boundary.py         The Q&A boundary-detection signature
    common.py                   Shared scoring scale and dimension metadata
scripts/                        Manual smoke-test scripts used during development
```

## Setup

Requires Python 3.12 and a Google Gemini API key.

```bash
uv venv --python 3.12 .venv
uv pip install -r requirements.txt --python .venv/bin/python
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your-key-here
```

## Running the app

```bash
.venv/bin/streamlit run app.py
```

Paste transcript text or upload a PDF in the sidebar, then click Analyze.

## Current status

Working end to end: transcript ingestion (paste or PDF), section splitting, six-dimension scoring, composite aggregation, and the two-pane UI with verified evidence highlighting.

Not yet built:

- **Optimization.** DSPy's `BootstrapFewShot` has not been run yet. The plan is to calibrate against a proxy metric derived from the dataset (change in forward-12-month EPS estimates between consecutive quarters for the same ticker), since the dataset does not contain a direct EPS-surprise field.
- **Price-reaction validation.** A secondary validation metric based on post-earnings stock price movement (via a market data API) is planned but not implemented.
- **Automated tests.** Current verification has been manual and script-based (see `scripts/`), not a pytest suite.

## Limitations

- The bundled dataset (`glopardo/sp500-earnings-transcripts`) covers 2013-05 to 2025-02 and is a static snapshot, not a live feed. It is used only for development calibration, not as an input source in the app itself.
- The regex fallback splitter was validated against that dataset's specific formatting and will not generalize as well as the primary LLM-based splitter to arbitrary transcript sources.
- Self-consistency reduces but does not eliminate scoring variance between runs on the same input.
- PDF text extraction does not handle scanned images without OCR.
