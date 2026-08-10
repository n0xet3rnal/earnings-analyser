"""Structured, serializable report combining pipeline scores + composite +
transcript metadata — the shape the UI (and any future API layer) consumes.

Evidence spans arrive relative to whichever section they were scored
against; this module shifts them to be relative to the *full* transcript
(`transcript_text`) so the UI can highlight evidence directly in the
original document regardless of which section it came from.
"""

from dataclasses import asdict, dataclass
from typing import Any

from .composite import compute_composite
from .data.loader import TranscriptRecord
from .modules.section_scorer import EvidenceQuote, SectionScores
from .pipeline import TranscriptScores
from .signatures import AGGREGATION_SIGN, DIMENSIONS


@dataclass(frozen=True)
class DimensionReportEntry:
    label: str
    raw_score: float  # in the dimension's own natural direction
    bullish_score: float  # sign-adjusted, used in composite math
    agreement: float
    rationale: str
    evidence: list[EvidenceQuote]  # start/end are absolute offsets into transcript_text


@dataclass(frozen=True)
class SectionReport:
    section_type: str
    bullish_score: float
    dimensions: dict[str, DimensionReportEntry]


@dataclass(frozen=True)
class TranscriptReport:
    ticker: str
    company: str
    sector: str
    earnings_date: str
    transcript_text: str
    composite_score: float
    split_confidence: str
    dimension_contributions: dict[str, float]
    sections: dict[str, SectionReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shift_evidence(evidence: list[EvidenceQuote], base_offset: int) -> list[EvidenceQuote]:
    return [
        EvidenceQuote(
            text=q.text,
            start=q.start + base_offset if q.start is not None else None,
            end=q.end + base_offset if q.end is not None else None,
        )
        for q in evidence
    ]


def _section_report(section: SectionScores, bullish_score: float, base_offset: int) -> SectionReport:
    dimensions = {
        dim: DimensionReportEntry(
            label=section.dimensions[dim].label,
            raw_score=section.dimensions[dim].score,
            bullish_score=AGGREGATION_SIGN[dim] * section.dimensions[dim].score,
            agreement=section.dimensions[dim].agreement,
            rationale=section.dimensions[dim].rationale,
            evidence=_shift_evidence(section.dimensions[dim].evidence, base_offset),
        )
        for dim in DIMENSIONS
    }
    return SectionReport(section_type=section.section_type, bullish_score=bullish_score, dimensions=dimensions)


def build_report(record: TranscriptRecord, scores: TranscriptScores) -> TranscriptReport:
    composite = compute_composite(scores)

    sections = {
        "prepared_remarks": _section_report(
            scores.prepared_remarks,
            composite.section_scores["prepared_remarks"],
            scores.prepared_remarks_offset,
        )
    }
    if scores.qa is not None:
        sections["qa"] = _section_report(scores.qa, composite.section_scores["qa"], scores.qa_offset)

    return TranscriptReport(
        ticker=record.ticker,
        company=record.company,
        sector=record.sector,
        earnings_date=record.earnings_date,
        transcript_text=scores.transcript_text,
        composite_score=composite.composite_score,
        split_confidence=scores.split_confidence,
        dimension_contributions=composite.dimension_contributions,
        sections=sections,
    )
