from typing import Literal

import dspy

from .common import ScoreLabel

SectionType = Literal["prepared_remarks", "qa"]


class DimensionScores(dspy.Signature):
    """Score this earnings call transcript section across 6 independent dimensions.
    For each dimension, choose exactly one of 5 discrete levels — strong_negative,
    mild_negative, neutral, mild_positive, strong_positive.

    For each dimension also provide:
    - evidence: 0-5 short quotes COPIED VERBATIM, word-for-word, from section_text
      (do not paraphrase, do not fix grammar/punctuation, do not add ellipses unless
      they appear in the source). Cite EVERY distinct passage that materially
      contributed to your judgment — if three separate statements each pushed you
      toward this label, quote all three, not just the single clearest one. Only
      use one quote if only one passage actually mattered; use an empty list if the
      signal comes from a genuinely diffuse pattern with no quotable passage — do
      not force a quote that isn't a real verbatim match just to fill this field.
    - rationale: 1-2 sentences explaining why the label fits. If evidence is empty,
      say so explicitly and describe the diffuse pattern instead of implying a quote
      exists.

    Score each dimension independently based only on the text; do not let your
    judgment on one dimension bias another.

    ## 1. Forward Guidance — how management handles forward guidance
    strong_positive: guidance raised, specific numeric targets, clear acceleration signals.
    mild_positive: guidance modestly positive or reaffirmed with an upward bias, no red flags.
    neutral: guidance reaffirmed in line with prior expectations, no notable change.
    mild_negative: guidance softened, narrowed, or vague without an outright cut.
    strong_negative: guidance cut or withdrawn, or evasive on numbers when pressed.

    ## 2. Uncertainty — degree of hedging, vagueness, and evasiveness
    strong_positive: pervasive hedging or evasiveness — repeated "too early to say," avoids specifics.
    mild_positive: noticeable hedging ("we believe," "we hope"), some analyst pushback on clarity.
    neutral: ordinary business-appropriate qualifiers, balanced between commitment and caution.
    mild_negative: mostly confident with only minor qualifiers.
    strong_negative: definitive, certain language throughout, clear commitments, no hedging.

    ## 3. Confidence — how assured and composed management sounds
    strong_positive: assured tone, direct answers, comfortable even under tough questions.
    mild_positive: generally confident with only minor moments of hesitation.
    neutral: typical corporate composure, neither notably assured nor hesitant.
    mild_negative: noticeable hesitation, some defensive or guarded answers.
    strong_negative: visibly defensive, flustered, or evasive under analyst pressure.

    ## 4. Sentiment — overall emotional tone and word choice (distinct from Confidence
    and Forward Guidance — this is about general linguistic valence and framing)
    strong_positive: enthusiastic, optimistic framing throughout ("thrilled," "record").
    mild_positive: generally positive tone, upbeat framing, analysts sound impressed.
    neutral: matter-of-fact tone, no strong emotional coloring.
    mild_negative: cautious or downbeat tone, some negative framing.
    strong_negative: overtly negative or pessimistic tone, visible analyst frustration.

    ## 5. Macro Focus — attribution of results/outlook to macro/external factors vs.
    company-specific execution
    strong_positive: results framed almost entirely around macro factors (rates, FX,
      geopolitics, industry-wide headwinds/tailwinds), little company-specific discussion.
    mild_positive: some macro framing alongside company-specific discussion.
    neutral: a balanced mix of macro and company-specific factors.
    mild_negative: mostly company-specific framing, macro mentioned only in passing.
    strong_negative: entirely company-specific framing (execution, product, strategy).

    ## 6. Jargon — density of corporate jargon/buzzwords vs. plain, concrete language
    strong_positive: dense with buzzwords/acronyms/vague corporate-speak ("synergies,"
      "leveraging our platform"), little concrete substance underneath.
    mild_positive: some jargon present but mostly clear.
    neutral: typical, moderate business language.
    mild_negative: mostly plain language with occasional technical terms.
    strong_negative: exceptionally clear, plain-spoken, concrete language throughout.
    """

    section_text: str = dspy.InputField(
        desc="Prepared remarks or Q&A section of an earnings call transcript"
    )
    section_type: SectionType = dspy.InputField(
        desc="Which part of the call this section is — affects expected structure/tone"
    )

    forward_guidance: ScoreLabel = dspy.OutputField()
    forward_guidance_evidence: list[str] = dspy.OutputField()
    forward_guidance_rationale: str = dspy.OutputField()

    uncertainty: ScoreLabel = dspy.OutputField()
    uncertainty_evidence: list[str] = dspy.OutputField()
    uncertainty_rationale: str = dspy.OutputField()

    confidence: ScoreLabel = dspy.OutputField()
    confidence_evidence: list[str] = dspy.OutputField()
    confidence_rationale: str = dspy.OutputField()

    sentiment: ScoreLabel = dspy.OutputField()
    sentiment_evidence: list[str] = dspy.OutputField()
    sentiment_rationale: str = dspy.OutputField()

    macro_focus: ScoreLabel = dspy.OutputField()
    macro_focus_evidence: list[str] = dspy.OutputField()
    macro_focus_rationale: str = dspy.OutputField()

    jargon: ScoreLabel = dspy.OutputField()
    jargon_evidence: list[str] = dspy.OutputField()
    jargon_rationale: str = dspy.OutputField()
