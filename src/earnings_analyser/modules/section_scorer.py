"""Scores one transcript section across all 6 dimensions.

Runs the combined DimensionScores signature `n_samples` times per section
(self-consistency) and takes a majority vote per dimension. This exists
because temperature=0 on Gemini reduces but does not eliminate run-to-run
drift (see scripts/test_signatures.py for the empirical finding) — running
a few samples at a small positive temperature and voting smooths that out,
and is cheap on the lite model we're using.

Every dimension result carries an audit trail: verbatim evidence quotes
(programmatically verified against the source text, not trusted on the
model's word) plus a rationale explaining the label — including cases
where the signal is diffuse across the section rather than one quotable
passage.
"""

import os
from collections import Counter
from dataclasses import dataclass

import dspy

from ..config import DEFAULT_MODEL
from ..signatures import DIMENSIONS, DimensionScores, SectionType, score_of
from ..text_matching import locate_span, normalize_text as _normalize

DEFAULT_N_SAMPLES = 3
SELF_CONSISTENCY_TEMPERATURE = 0.3


@dataclass(frozen=True)
class EvidenceQuote:
    text: str
    start: int | None  # offset within the scored section_text; None if unlocatable
    end: int | None

    @property
    def verified(self) -> bool:
        return self.start is not None


@dataclass(frozen=True)
class DimensionResult:
    label: str
    score: float  # mean of mapped anchor values across samples
    agreement: float  # fraction of samples that agreed with the majority label
    rationale: str  # from the first sample that matched the majority label
    evidence: list[EvidenceQuote]  # union of evidence across ALL agreeing samples, deduplicated


@dataclass(frozen=True)
class SectionScores:
    section_type: SectionType
    dimensions: dict[str, DimensionResult]


class SectionScorer(dspy.Module):
    def __init__(self, n_samples: int = DEFAULT_N_SAMPLES):
        super().__init__()
        self.n_samples = n_samples
        self.predict = dspy.Predict(DimensionScores)
        self._sample_lm = dspy.LM(
            DEFAULT_MODEL,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=SELF_CONSISTENCY_TEMPERATURE,
            cache=False,
        )

    def forward(self, section_text: str, section_type: SectionType) -> SectionScores:
        with dspy.context(lm=self._sample_lm):
            samples = [
                self.predict(section_text=section_text, section_type=section_type)
                for _ in range(self.n_samples)
            ]

        dimensions = {}
        for dim in DIMENSIONS:
            labels = [getattr(s, dim) for s in samples]
            rationales = [getattr(s, f"{dim}_rationale") for s in samples]
            evidence_lists = [getattr(s, f"{dim}_evidence") for s in samples]
            majority_label, majority_count = Counter(labels).most_common(1)[0]

            # Union evidence across every sample that agreed with the
            # majority label, not just one representative sample — different
            # samples often surface different, equally valid supporting
            # passages even when they land on the same label, so taking only
            # one understates what actually drove the judgment. Dedup by
            # located span (not raw text) where possible: two samples
            # citing the same real sentence with minor wording drift both
            # resolve to the same (start, end) via locate_span's fuzzy
            # matching, so this collapses them correctly; quotes that don't
            # verify at all fall back to a normalized-text dedup key.
            rationale = None
            evidence: list[EvidenceQuote] = []
            seen = set()
            for label, rationale_text, quotes in zip(labels, rationales, evidence_lists):
                if label != majority_label:
                    continue
                if rationale is None:
                    rationale = rationale_text
                for quote in quotes:
                    if not quote.strip():
                        continue
                    span = locate_span(quote, section_text)
                    dedup_key = span if span is not None else _normalize(quote)
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)
                    evidence.append(
                        EvidenceQuote(text=quote, start=span[0] if span else None, end=span[1] if span else None)
                    )

            dimensions[dim] = DimensionResult(
                label=majority_label,
                score=sum(score_of(label) for label in labels) / len(labels),
                agreement=majority_count / len(labels),
                rationale=rationale,
                evidence=evidence,
            )

        return SectionScores(section_type=section_type, dimensions=dimensions)
