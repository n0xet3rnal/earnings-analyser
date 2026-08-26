"""Structured report the UI consumes: per dimension, the terminal
conclusions produced by the collapse (implementation-plan.md §2.3), each
with its own label, narrative, and cited evidence.

Evidence is always a source sentence resolved by id through the graph
store — never re-quoted or reproduced by a model at this stage (see
`analysis/attribution.py`'s docstring) — with the attribution weight
attached so the UI can render the heatmap directly from this shape.
"""

from dataclasses import asdict, dataclass
from typing import Any

from .analysis.attribution import compute_attribution
from .persistence.graph_store import GraphStore
from .signatures import DIMENSIONS

DEFAULT_EVIDENCE_TOP_K = 8


@dataclass(frozen=True)
class EvidenceEntry:
    text: str
    start: int
    end: int
    weight: float  # this conclusion's attribution weight for this sentence


@dataclass(frozen=True)
class Conclusion:
    label: str | None
    narrative: str
    evidence: list[EvidenceEntry]


@dataclass(frozen=True)
class TranscriptReport:
    ticker: str
    company: str
    earnings_date: str
    transcript_text: str
    conclusions: dict[str, list[Conclusion]]  # dimension -> terminal conclusions (themes)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_report(
    store: GraphStore,
    transcript_text: str,
    ticker: str,
    company: str,
    earnings_date: str,
    evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
) -> TranscriptReport:
    conclusions: dict[str, list[Conclusion]] = {}

    for dim in DIMENSIONS:
        attribution = compute_attribution(store, dim)
        dim_conclusions = []

        for terminal in store.terminal_nodes(dim):
            leaf_weights = attribution.get(terminal.node_id, {})
            top = sorted(leaf_weights.items(), key=lambda kv: kv[1], reverse=True)[:evidence_top_k]

            evidence = []
            for sentence_id, weight in top:
                sentence = store.get_node(sentence_id)
                if sentence is None or sentence.start is None or sentence.end is None:
                    continue
                evidence.append(EvidenceEntry(text=sentence.text, start=sentence.start, end=sentence.end, weight=weight))

            dim_conclusions.append(Conclusion(label=terminal.label, narrative=terminal.text, evidence=evidence))

        conclusions[dim] = dim_conclusions

    return TranscriptReport(
        ticker=ticker,
        company=company,
        earnings_date=earnings_date,
        transcript_text=transcript_text,
        conclusions=conclusions,
    )
