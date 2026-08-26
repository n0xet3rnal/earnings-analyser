# Earnings Call Analyzer

Analyzes earnings call transcripts across six linguistic dimensions drawn from academic research on analyst communication. Runs entirely locally against a small open-weight model via Ollama — no cloud API key required. Every conclusion cites specific, located sentences from the transcript; nothing is asserted without a traceable source.

## Academic basis

**Matera 2024** ([arXiv:2511.15214](https://arxiv.org/abs/2511.15214)) defines the six dimensions used here: Forward Guidance, Uncertainty, Confidence, Sentiment, Macro Focus, and Jargon.

**"Fast Numbers, Slow Language"** ([arXiv:2606.29734](https://arxiv.org/abs/2606.29734)) informed the discrete five-anchor label scale (`strong_negative`..`strong_positive`) used per theme.

There is no numeric composite score. Earlier iterations of this project scored each dimension as an average float and combined the six into a single weighted number — those weights were never validated against anything, and averaging LLM-classified labels manufactures precision that isn't there. This version replaces that with grounded, cited narrative themes plus a coarse label per theme.

## What it does

1. **Sentence split.** The transcript is deterministically split into sentences, each with a real, absolute offset into the source text — no model involved.
2. **Base grouping.** Sentences are grouped into small overlapping windows (~4-5 sentences, sharing one sentence with each neighbor).
3. **Recursive collapse.** One call per group produces, for all six dimensions at once: a relevance weight for each input sentence/composite, and a short grounded composite passage. The same call is reused recursively — composites from one round become inputs to the next — until each dimension is down to a handful (~4-5) of terminal themes, each carrying a coarse label.
4. **Deterministic attribution.** After the collapse finishes, a pure-code pass (no model call) chains every round's relevance weights together to compute exactly how much each original sentence contributed to each final theme — a real, computable importance score, not just "cited or not."
5. **UI.** Streamlit shows each dimension's themes as cards (label + narrative); clicking one renders the full transcript with a heatmap — sentence highlight intensity proportional to that sentence's computed importance to the selected theme.

## Why this shape

- **Sentence-level, not whole-section.** A single call judging six dimensions over an entire ~20k-word section is the wrong task for a small local model. Judging six dimensions over 4-5 sentences at a time is a much easier, more reliable task — and the same signature is reused recursively at every level, so the total interface stays small.
- **Classification, not floats.** Both the relevance weights (`none/weak/moderate/strong`) and the per-theme labels are discrete classifications. Local models are reliable classifiers; they are not reliable at inventing calibrated numbers, and nothing in this pipeline asks them to.
- **Citations by construction, not by verification.** The model never reproduces transcript text — it references inputs by index. There is nothing to hallucinate or fuzzy-match against the source, because nothing above the leaf (sentence) layer ever claims to *be* source text.
- **One label per theme, not one per dimension.** Aggregating several themes' labels into a single dimension-level verdict reintroduces exactly the majority-vote-with-ties problem an earlier version of this codebase had (label and score silently disagreeing). Each theme keeps its own label instead.

## Tech stack

- **DSPy** for structured signatures — kept specifically because a future prompt-optimization pass (a larger model generating training data, `BootstrapFewShot` compiling it into few-shot demonstrations for the local model) depends on it directly.
- **Ollama**, serving a local model (default: Qwen3-4B-Instruct) through an OpenAI-compatible endpoint. Benchmarked against Qwen2.5-7B-Instruct Q4_K_M with caching disabled and identical terse-summary prompts on both: Qwen3-4B ran ~25-30% faster (~33s vs. ~43-45s on a 12-sentence test transcript) thanks to a real per-token speed advantage (65.6 tok/s vs. 31.8 tok/s measured directly against Ollama).
- **SQLite (WAL mode)** as the graph store — nodes and edges for every collapse level, written incrementally so a live viewer can read the graph while the pipeline is still building it, and so an interrupted run resumes instead of restarting.
- **Streamlit** for the UI, **Plotly**-free now (no composite score to chart); **pypdf** for PDF transcript extraction.

## Project structure

```
app.py                          Streamlit UI — theme cards + evidence heatmap
src/earnings_analyser/
  config.py                     DSPy + local Ollama configuration
  pipeline.py                   Wires a live predictor to the collapse orchestration
  report.py                     Builds the {dimension: [themes]} shape the UI consumes
  data/
    sentence_split.py           Deterministic sentence split + overlapping base windows
    pdf_extract.py               PDF to text extraction
  signatures/
    dimension_collapse.py       The one recurring signature: 6-dimension relevance + summary + (terminal) label
    common.py                    Shared label scale and dimension list
  modules/
    collapse_step.py            One level of the recursion, and the full orchestration loop
  analysis/
    attribution.py               Pure-code weighted-path attribution (no model calls)
  persistence/
    graph_store.py               SQLite/WAL-backed node/edge store, resumable
scripts/
  smoke_test.py                  Manual check that the local Ollama endpoint is reachable
tests/                           pytest coverage for the deterministic parts (offsets, graph store, attribution math)
```

## Setup

Requires Python 3.12 and a local [Ollama](https://ollama.com) install.

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve   # if not already running

python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Use `.venv/bin/python -m <tool>` rather than `.venv/bin/<tool>` directly — a venv's console scripts hardcode their creation path in the shebang line, so `.venv/bin/streamlit` (etc.) breaks with "file not found" if the project directory is ever moved or renamed after the venv was created; invoking through `python -m` doesn't have that problem.

## Running the app

```bash
.venv/bin/python -m streamlit run app.py
```

Paste transcript text or upload a PDF in the sidebar, click Analyze, then pick a dimension and click a theme card to see its evidence heatmap.

## Running tests

```bash
.venv/bin/python -m pytest tests/
```

All current tests are pure — no network or live model required. They cover sentence-split offset correctness (including window overlap), the graph store's read/write/resumability behavior, and the attribution math (single-level weights, multi-level chains, and multi-parent convergence at overlapping window boundaries).

## Current status

Implemented: sentence split, overlapping base windows, the merged collapse signature, the recursive collapse orchestration with resumability, deterministic attribution, the SQLite/WAL graph store, and the Streamlit UI (theme cards + heatmap).

Not yet done:

- **Empirical tuning.** Branching factor per collapse round, and behavior on very short transcripts, are set to reasonable defaults but not yet tuned against real transcripts.
- **Concurrency's real ceiling.** Client-side thread-pool concurrency for independent collapse calls is implemented, but measured only a ~4% wall-clock improvement on this hardware (6GB GPU, model already using ~4.77GB VRAM) — Ollama appears to serialize requests server-side regardless of client concurrency here. The code is correct and harmless but isn't the win it might look like on paper; call-count reduction (branching factor, window size) matters more than parallelism on this setup.
- **DSPy prompt optimization.** A separate, not-yet-started workstream: generate training data with a larger teacher model, compile it into few-shot demonstrations for the local model via `BootstrapFewShot`.
- **Live graph visualization.** The graph store's SQLite/WAL design supports a concurrent reader while the pipeline writes, but no viewer has been built yet.

See `implementation-plan.md` for the full architecture writeup and the reasoning behind each decision.
