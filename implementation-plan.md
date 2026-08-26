# Earnings Analyser — Implementation Plan

**Status:** confirmed, implementing
**Owner:** Jerry
**History:** merges `changes.md` (local-model migration, hardware sizing) + `build-plan.md` v0.2 (recursive weighted-collapse architecture); both superseded and deleted. Architecture decisions below were walked through and confirmed one at a time (see §2-§5). A ponytail-audit pass (over-engineering only) proposed 8 cuts; #1-2 (simplifying to a fixed 2-level collapse, and dropping SQLite for plain JSON) were explicitly rejected — the general N-level recursion and the SQLite/WAL graph store are kept. Findings #3-8 were accepted and are folded in below.
**Out of scope:** DSPy prompt optimization / training-data generation — a separate, parallel workstream.

**Post-implementation update (performance):** a 1000-sentence transcript projected to ~267 LLM calls / ~34 minutes at the original defaults, confirmed by live-measured per-call latency, not just estimated — not tractable. A more ambitious fix (self-directed clustering: read a much bigger batch per call, let the model group items into a few themes and pick representatives, cutting call count by decoupling output size from input size) was designed, probed live, and — after a real `BootstrapFewShot` optimization attempt against both models with a corrected metric (Adjusted Rand Index; the first metric used, plain pairwise agreement, was itself found to be broken — a degenerate non-clustering prediction scored ~0.68 on it) — rejected. Best result: Qwen2.5-7B compiled reached ARI 0.127, more than double its zero-shot 0.052, but still far below a trustworthy clustering signal (1.0 = perfect, 0 = chance). Qwen3-4B stayed at chance level even after optimization (-0.004 → 0.019). See `scripts/generate_segmenter_data.py`, `scripts/optimize_segmenter.py`, `src/earnings_analyser/signatures/segmenter.py`, `src/earnings_analyser/analysis/segmenter_metric.py` for the full experiment; not wired into the live pipeline.

Reverted to the cheaper, already-proven levers instead: base `window_size` 5→8, `window_overlap` 1→**0**, `DEFAULT_BRANCHING_FACTOR` 8→16 (already applied earlier), plus concurrent dispatch of independent calls within a level (real but modest win on this hardware — see `README.md`). Dropping overlap to 0 is a real architectural reversal, not just a tuning knob: overlap was the *only* source of multi-parent structure in the collapse graph (§2.3/§2.4 below); the live graph is now a strict tree end-to-end. `overlap` remains a supported parameter in `data/sentence_split.py` if boundary-split evidence turns out to matter more than call count in practice. Redundancy/near-duplicate deduplication (embeddings-based, previously discussed as a complementary lever) was deliberately not pursued alongside this — held for later.

---

## 1. Context

The original build was a single-shot Gemini pipeline: one API call scored an entire transcript section across six dimensions as calibrated floats, three samples got majority-voted, and the votes fed a hand-weighted composite. Three separate problems drove this rebuild: it required a cloud API key, a single call over ~20k words asking for six independent judgments was the wrong task shape for a local model, and the composite's weights were never validated against anything.

The resolution: process at **sentence granularity**, recursively collapse sentence-level evidence into a handful of grounded conclusions per dimension, and replace the composite with cited narrative + a coarse label. Every claim traces back to real transcript text, and every sentence gets a deterministic, computable importance score, not just a binary "cited or not" — which is what makes the heatmap requirement (see §2.5) possible at all.

### Goals

- **Runs fully offline** via Ollama, no cloud API key required for normal operation.
- **Long transcripts don't need one context window.** Bounded, resumable processing.
- **Citations are load-bearing.** Every claim traces to a verified, located sentence in the original transcript.
- **No numeric composite.** A coarse per-dimension label is kept; nothing gets averaged into a false-precision float.
- **Resumable, and eventually watchable.** Long local-model runs persist results as they complete; the graph store is built to support a live viewer reading while the pipeline writes (see §5).
- **Actually tested.** The pure, deterministic parts (offset math, attribution matrix math, graph store) get real unit tests.

---

## 2. Architecture: recursive weighted collapse

### 2.1 One function, six dimensions bundled

Every level of processing — from raw sentences up to the final handful of conclusions — is built from **one recurring operation**, applied recursively regardless of whether the inputs are raw transcript sentences or composites from a previous level: given a small group of indexed inputs, produce, for all six dimensions in a single call:

- an integer relevance score 0-3 per input, per dimension (0=irrelevant..3=central) — an int, not a word bucket. An earlier version used a 4-word vocabulary and a live run against Qwen2.5-7B showed the model occasionally leaking a word from the `ScoreLabel` vocabulary (`"neutral"`) into a weight field, crashing strict parsing. A bare int shares no vocabulary with anything, so there's nothing to leak;
- a short composite passage per dimension, grounded in whichever inputs were weighted highest;
- (terminal invocations only) a coarse `ScoreLabel` per dimension.

This was originally split into two functions — a relevance-checker and a summarizer — called separately. They're merged: both already bundle all six dimensions into one call each (§2.2's reasoning — one text span, multiple independent output slots, is safe to bundle), and that same reasoning applies to bundling the two *functions* together, since judging relevance and writing the grounded summary are both single-pass reads of the same small group. One call per group per level, not two — this halves the total LLM-call volume of the whole pipeline. The signature (`signatures/dimension_collapse.py`) is reused unchanged at every level, so the system stays genuinely recursive.

### 2.2 Call batching: what's batched, what isn't, and why

Two different things could be called "batching," with very different risk:

- **Bundling all six dimensions into one call, for one group of text** — safe. The model reads one small passage once and fills in six independent, labeled output slots.
- **Bundling multiple groups (different text spans) into one call** — unsafe. Nothing stops the model attributing evidence across spans that were never meant to interact, and it breaks the attribution math in §2.4, which needs each level's weights to be a clean distribution over one group's own inputs.

**Rule: one call per group, six dimensions (and now both judgment + synthesis) bundled inside it.**

### 2.3 Base grouping and recursive collapse

1. **Sentence split.** Deterministic, in code, no model involved — every sentence gets a real offset into `transcript_text`.
2. **Base grouping.** Windows of ~4-5 sentences. **Adjacent windows overlap by 1 sentence** — each window's last sentence is also the next window's first. This is what creates the only multi-parent edges in the graph: a leaf sentence at a window boundary gets weighted by two separate calls and can end up cited by two different level-1 composites, which is exactly the non-tree connectivity the design calls for, and it's also what prevents a claim/explanation pair split across a boundary from being lost.
3. **Collapse.** Group the previous level's composites using a **fixed branching factor per round** (value set empirically — see §9), not a single upfront-computed divisor — this keeps every call's cognitive load constant regardless of transcript length; the number of rounds adapts instead. Every level above the base grouping is a strict partition (each composite has exactly one parent) — composites are already synthesized, coherent text by then, not raw fragments that can be arbitrarily split, so a strict partition is the right structure there.
4. **Stop** when a dimension is down to ~4-5 composites. Depth is not fixed — the loop runs as many rounds as a given transcript actually needs.
5. **Terminal labeling.** Only at the stopping point, the merged call also fills the `ScoreLabel` field per terminal composite, **one label per theme** — not aggregated into a single label per dimension. Aggregating would reintroduce the majority-vote-with-ties problem that was the first bug found in the original codebase (label and score disagreeing after a 3-way tie).

### 2.4 Deterministic attribution

Each level's relevance weights form a matrix: children (this level's inputs) × parents (this level's composites), normalized so each parent's weights over its own inputs sum to 1. A leaf sentence's total importance to a final conclusion is the product of edge weights along every path from that leaf to that conclusion, summed across all paths — matrix multiplication: chain every level's weight matrix together (`W1 @ W2 @ ... @ Wk`, k rounds, k determined per-transcript by §2.3) and read off the leaf-to-conclusion entries.

Pure code, no model call, run once after all collapse levels are done. Multi-parent convergence (the "sum across paths" term) is real specifically at the level-0→1 boundary via the 1-sentence window overlap (§2.3.2); every level above is a strict partition by construction, so a leaf above level 1 has exactly one path forward — expected, not a gap.

**Citation granularity is single sentence only, never a span.** This was decided specifically because it matches the heatmap requirement: every leaf gets its own attribution weight, so intensity-per-sentence is exact rather than approximated across a multi-sentence span.

### 2.5 Heatmap rendering

The final attribution matrix — one column per terminal conclusion — is a real number per leaf sentence per conclusion. Pick a conclusion in the UI, color each sentence by its value in that column. Same click-to-highlight interaction the original `app.py` had; continuous intensity instead of a binary mark.

---

## 3. Local model serving boundary

Fix the original ordering bug: modules must take the LM via `dspy.configure` / `dspy.context` only, never construct their own — the original code had each module construct its own `dspy.LM` and read `GEMINI_API_KEY` directly, silently overriding `configure_dspy()`'s settings and raising a bare `KeyError` if a module was touched before configuration.

**Serving layer: Ollama**, not a raw llama.cpp server — lower friction to stand up and switch models, OpenAI-compatible endpoint out of the box, and this is a resume-facing rebuild where working, demonstrable behavior matters more than squeezing out serving-layer performance.

**Default model: Qwen2.5-7B-Instruct Q4_K_M** (RTX 3060 Laptop, 6GB VRAM, ~5.1GB free at measurement time), kept as the day-one default. **Qwen3-4B-Instruct** is benchmarked as a comparison once the pipeline is working — not blocking early phases, since per-call context is now tiny (a group of ~4-5 sentences or composites) rather than a whole section, which is a meaningfully easier task than the original sizing assumed. Disable Qwen3 thinking mode for structured DSPy outputs unless testing shows it helps.

DSPy and its signature abstraction are kept deliberately — not just inertia: the DSPy-optimization workstream (training data generated by a larger teacher model, `BootstrapFewShot` compiling few-shot demonstrations into the local model's prompts) depends on it directly, so the abstraction has a real, committed near-term use, unlike speculative flexibility.

---

## 4. Component changes

Legend: **Keep** · **Modify** · **Remove** · **New**

| File | Action | Notes |
|---|---|---|
| `text_matching.py` | Remove | Turned out to be genuinely dead once implemented: its fuzzy verified-span matching existed to check a model's *reproduction* of source text. In the built pipeline, the model never reproduces text — relevance weights reference inputs by index (`dimension_collapse.py`), and sentence offsets come from `sentence_split.py`'s deterministic split, not from anything the model outputs. Confirmed by grep: nothing in the built codebase imports it. |
| `data/sentence_split.py` | New | Deterministic sentence tokenizer + base-window grouping (~4-5 sentences, 1-sentence overlap between adjacent windows). Every leaf gets a real, absolute offset into `transcript_text`, computed directly from the split — no fuzzy matching needed since nothing here is verifying a model's reproduction of anything. |
| `signatures/dimension_collapse.py` | New | The merged signature (§2.1). Six dimension rubrics (ported from the original `dimension_scores.py`), bundled in one call: per-dimension relevance weights + composite text, plus a `ScoreLabel` field used only at terminal invocations. |
| `signatures/dimension_scores.py` | Remove | Superseded — no more single-shot six-dimension classification over a whole section, and no more separate relevance/summary calls either. |
| `signatures/section_boundary.py` | Remove | Unused — sentence-level processing has no section boundary to find. |
| `data/splitter.py` | Remove | Only existed to back `section_boundary.py`. |
| `modules/section_scorer.py` | Remove | Self-consistency voting no longer needed. |
| `modules/collapse_step.py` | New | Implements *one level* of the recursion: one call per group via `DimensionCollapse`, writes new composite nodes + that level's weight matrix to the graph store. Reused for every level including the terminal one. |
| `analysis/attribution.py` | New | Pure code, no model calls. Chains stored per-level weight matrices into leaf-to-conclusion importance scores. |
| `persistence/graph_store.py` | New | SQLite, WAL mode. See §5. One writer (the pipeline) inserting nodes/edges as they're produced; supports concurrent readers for a future live graph viewer. |
| `pipeline.py` | Modify | Orchestration: sentence-split → base-group → loop `collapse_step` until stop condition → terminal labeling → `attribution.py` → report. |
| `composite.py` | Remove | No replacement. |
| `report.py` | Modify | Per dimension → list of terminal conclusions, each `{label, narrative, evidence: [...]}`, evidence entries carrying the cited sentence + its attribution weight. |
| `config.py` | Modify | Fix the ordering bug; one Ollama-backed OpenAI-compatible config, `DEFAULT_MODEL = "qwen2.5:7b-instruct-q4_K_M"`. |
| `app.py` | Modify | Conclusion cards (label + narrative) per theme, heatmap-intensity transcript pane instead of binary highlighting. |
| `data/loader.py` | Remove (archived) | Moved out of the active package — nothing in the new pipeline calls it. It only backs the parked, unscoped EPS-revision validation idea; revive from history if that's picked up later. |
| `tests/` | New | Unit coverage for `sentence_split.py`'s offset math (incl. the 1-sentence overlap), `graph_store.py`'s read/write + resumability, and `attribution.py`'s matrix chaining — all pure, no network or live model needed. |
| `scripts/smoke_test.py` | Modify | Trimmed to one manual check: is the configured Ollama endpoint reachable and does a bare DSPy call round-trip. |
| `scripts/test_composite.py`, `scripts/test_pipeline.py`, `scripts/test_signatures.py` | Remove | Redundant with real pytest coverage in `tests/`; were manual, duplicated `sys.path` + `configure_dspy()` + `load_dataset()` boilerplate around what should be actual tests. |
| `pyproject.toml` | New | Replaces `sys.path.insert` hacks duplicated across `app.py` and all of `scripts/*.py` with a proper installable package. |
| `requirements.txt` | Modify | Drop `google-generativeai` (unused — litellm talks to Gemini's `gemini/` provider directly, and that path is now build-time-only for the optimizer workstream anyway) and `datasets`/`huggingface-hub`/`pandas` (only used by the now-archived `loader.py`). Add an Ollama-compatible client dependency. |

---

## 5. Data model

Graph store, SQLite with WAL mode, one database per transcript (or one DB with a `transcript_id` column if a single-file-per-run store turns out to be awkward — decide when `graph_store.py` is actually written). WAL mode specifically because a live graph viewer is expected to read while the pipeline is still writing; it's built for exactly that one-writer/concurrent-readers pattern.

```sql
CREATE TABLE nodes (
    node_id     TEXT PRIMARY KEY,
    level       INTEGER NOT NULL,
    kind        TEXT NOT NULL,       -- 'source_sentence' | 'composite'
    dimension   TEXT,                -- NULL for source_sentence (shared across dimensions)
    start_off   INTEGER,             -- absolute offset into transcript_text; source_sentence only
    end_off     INTEGER,
    text        TEXT NOT NULL,
    label       TEXT,                -- NULL until terminal
    terminal    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE edges (
    child       TEXT NOT NULL REFERENCES nodes(node_id),
    parent      TEXT NOT NULL REFERENCES nodes(node_id),
    dimension   TEXT NOT NULL,
    weight      REAL NOT NULL,       -- normalized within (parent, dimension)
    PRIMARY KEY (child, parent, dimension)
);

CREATE TABLE run_status (
    level       INTEGER NOT NULL,
    complete    INTEGER NOT NULL     -- resumability: last fully-written level
);
```

`analysis/attribution.py` reads `edges`, reshapes them per dimension into per-level matrices, and chains them to produce `{terminal_node_id: {source_sentence_id: importance}}` — the table the heatmap renders directly.

---

## 6. Carried-over fixes

- **Config/env ordering coupling** — §3.
- **Unused/dormant dependencies** — `google-generativeai`, and `datasets`/`huggingface-hub`/`pandas` once `loader.py` is archived — §4.
- **Zero tests** — §4.
- **`sys.path.insert` hacks** — `pyproject.toml`, §4.
- **Duplicated manual test scripts** — consolidated into `tests/`, §4.

---

## 7. Migration acceptance criteria

- No Gemini API key is required for normal local operation.
- A transcript can be processed incrementally — level by level, group by group.
- Completed levels survive interruption and are reused on resume, not recomputed.
- Every claim in the final report cites a real, located sentence in the original transcript.
- Every leaf sentence has a computable, deterministic importance score per terminal conclusion.
- Evidence offsets refer to the original full transcript, not to intermediate composite text.
- The graph store supports a concurrent reader while the pipeline is still writing (even before a live viewer is built, this is a testable property of `graph_store.py`).

---

## 8. Suggested phasing

1. **Sentence split + graph store** — `data/sentence_split.py`, `persistence/graph_store.py` (SQLite/WAL), full unit coverage. No model calls yet.
2. **Local serving boundary** — fix `config.py`, wire Ollama, confirm a bare DSPy call round-trips against Qwen2.5-7B-Instruct Q4_K_M.
3. **`DimensionCollapse` signature** — validate against real base-window groups: does the merged one-call-per-group (weights + summary + six dimensions) actually hold up.
4. **`collapse_step.py` + orchestration loop** — wire the recursion, confirm it converges to ~4-5 terminal composites per dimension on a real transcript, and use that run to pick the branching factor.
5. **`attribution.py`** — pure-code matrix chaining.
6. **Report + UI** — conclusion cards, heatmap rendering.
7. **Packaging + scripts** — `pyproject.toml`, trim `requirements.txt`, archive `data/loader.py`, cut the redundant manual scripts.

---

## 9. Open questions for architecture review

- **Branching factor per collapse round** — set empirically from real transcripts' base-window counts (phase 4).
- **Overlap behavior on very short transcripts** — e.g. a call with only 1-2 base windows total; the 1-sentence-overlap rule needs a defined edge case there.
- **Qwen2.5-7B vs. Qwen3-4B** — resolve via the phase-2/3 benchmark, not upfront.
- **Live-visualization frontend** — SQLite/WAL was chosen to support a future live graph viewer, but how it's actually rendered (a Streamlit auto-refresh panel vs. a separate small viewer) hasn't been decided.
- **One DB file per transcript vs. one shared DB with a `transcript_id` column** — decide when `graph_store.py` is written (§5).
