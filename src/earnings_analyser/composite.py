"""Combine per-dimension, per-section scores into a single composite.

Two academic findings drive the weighting:
  - Matera 2024: analysts underweight uncertainty, so it gets upweighted
    relative to the other dimensions.
  - "Fast Numbers, Slow Language": Q&A is more signal-rich than prepared
    remarks, so it gets more weight in the section blend.

Weights are plain dicts passed as arguments (not hardcoded constants deep
in the logic) so they can be swapped out — e.g. by BootstrapFewShot-style
tuning later, or just manual experimentation.
"""

from dataclasses import dataclass

from .modules.section_scorer import SectionScores
from .pipeline import TranscriptScores
from .signatures import AGGREGATION_SIGN, DIMENSIONS

DEFAULT_DIMENSION_WEIGHTS: dict[str, float] = {
    "forward_guidance": 1.0,
    "uncertainty": 2.0,  # upweighted per Matera 2024
    "confidence": 1.0,
    "sentiment": 1.0,
    "macro_focus": 0.75,
    "jargon": 0.75,
}

DEFAULT_SECTION_WEIGHTS: dict[str, float] = {
    "prepared_remarks": 0.4,
    "qa": 0.6,  # weighted higher per Fast Numbers, Slow Language
}


@dataclass(frozen=True)
class CompositeResult:
    composite_score: float  # -1..+1, bullish-signed
    section_scores: dict[str, float]  # bullish-signed score per section
    dimension_contributions: dict[str, float]  # bullish-signed score per dimension, blended across sections


def _section_bullish_score(
    section: SectionScores, dimension_weights: dict[str, float]
) -> tuple[float, dict[str, float]]:
    """Returns (weighted section score, per-dimension bullish-signed scores)."""
    per_dim = {
        dim: AGGREGATION_SIGN[dim] * section.dimensions[dim].score for dim in DIMENSIONS
    }
    total_weight = sum(dimension_weights[dim] for dim in DIMENSIONS)
    weighted = sum(per_dim[dim] * dimension_weights[dim] for dim in DIMENSIONS) / total_weight
    return weighted, per_dim


def compute_composite(
    scores: TranscriptScores,
    dimension_weights: dict[str, float] = DEFAULT_DIMENSION_WEIGHTS,
    section_weights: dict[str, float] = DEFAULT_SECTION_WEIGHTS,
) -> CompositeResult:
    sections = {"prepared_remarks": scores.prepared_remarks}
    if scores.qa is not None:
        sections["qa"] = scores.qa

    section_scores: dict[str, float] = {}
    dim_by_section: dict[str, dict[str, float]] = {}
    for name, section in sections.items():
        section_scores[name], dim_by_section[name] = _section_bullish_score(
            section, dimension_weights
        )

    # Renormalize section weights over whichever sections are actually present
    # (e.g. transcripts where the Q&A split failed fall back to prepared-remarks-only).
    active_weights = {name: section_weights[name] for name in sections}
    total_section_weight = sum(active_weights.values())
    composite_score = sum(
        section_scores[name] * active_weights[name] for name in sections
    ) / total_section_weight

    dimension_contributions = {
        dim: sum(
            dim_by_section[name][dim] * active_weights[name] for name in sections
        )
        / total_section_weight
        for dim in DIMENSIONS
    }

    return CompositeResult(
        composite_score=composite_score,
        section_scores=section_scores,
        dimension_contributions=dimension_contributions,
    )
