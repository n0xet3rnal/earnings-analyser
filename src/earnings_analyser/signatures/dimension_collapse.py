import dspy

from .common import ScoreLabel

# Integer relevance score, not a word bucket: an earlier version used a
# 4-word vocabulary (irrelevant/minor/notable/central) and a 7B model
# still occasionally leaked "neutral" from the *_label vocabulary into a
# *_weights field. A bare int has no vocabulary to confuse with anything.
# 0-3 is deliberately a small fixed range, not an open float — still easy
# to validate/clamp, unlike asking the model to invent a calibrated
# continuous number. Normalized in `modules/collapse_step.py`.
MAX_RELEVANCE_SCORE = 3

# Shared by both signatures below — the rubric doesn't change based on
# whether inputs are raw sentences or per-dimension composites, only the
# input field shape does (see DimensionCollapseBundled).
_RUBRIC = """\
Score each dimension independently, based only on its own inputs — do not
let your judgment on one dimension bias another, and do not reuse content
from one dimension's inputs when scoring or summarizing a different one.

For each of the six dimensions below, do two things:

1. Judge how relevant each input is to that dimension. Output a
   list of integers, one per input IN THE SAME ORDER as the inputs
   (not a mapping, a plain ordered list — length must match the
   number of inputs exactly): 0 = irrelevant, 1 = minor, 2 = notable,
   3 = central. Integers only — never a word, never a decimal.
2. Write ONE sentence, 25 words max, for that dimension's composite,
   grounded in whichever inputs you scored 1 or higher (ignore
   0-scored inputs) — this becomes one input to the next collapse
   round, or, if `is_terminal` is true, the final grounded conclusion
   for this theme. Do not pad, hedge, or add a second sentence.

If `is_terminal` is true, also classify the dimension's overall
reading for this group as an integer 1-5: 1=strong_negative,
2=mild_negative, 3=neutral, 4=mild_positive, 5=strong_positive. This
is a coarse classification of the composite you just wrote, not a new
judgment pass over the raw inputs. Integer only, never a word. Leave
the label fields null when `is_terminal` is false.

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

_LABEL_DESC = "1-5 (1=strong_negative..5=strong_positive), or null if not terminal"
_SUMMARY_DESC = "ONE sentence, 25 words max, no padding"


class DimensionCollapse(dspy.Signature):
    __doc__ = (
        "Given a small, indexed group of inputs (either raw transcript "
        "sentences or composite passages from a previous collapse round):\n\n" + _RUBRIC
    )

    group_items: list[str] = dspy.InputField(
        desc="Indexed inputs for this group, formatted as '[<index>] <text>' one per line"
    )
    is_terminal: bool = dspy.InputField(
        desc="True if this group's composite is a final conclusion, not input to a further round"
    )

    forward_guidance_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Forward Guidance — length must equal the number of inputs"
    )
    forward_guidance_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    forward_guidance_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    uncertainty_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Uncertainty — length must equal the number of inputs"
    )
    uncertainty_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    uncertainty_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    confidence_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Confidence — length must equal the number of inputs"
    )
    confidence_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    confidence_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    sentiment_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Sentiment — length must equal the number of inputs"
    )
    sentiment_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    sentiment_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    macro_focus_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Macro Focus — length must equal the number of inputs"
    )
    macro_focus_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    macro_focus_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    jargon_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per input, IN ORDER (0=irrelevant, 3=central), for Jargon — length must equal the number of inputs"
    )
    jargon_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    jargon_label: int | None = dspy.OutputField(desc=_LABEL_DESC)


class DimensionCollapseBundled(dspy.Signature):
    __doc__ = (
        "Used from the second collapse round onward, once composites have "
        "already diverged per dimension — each dimension gets its OWN "
        "separate input list below instead of one shared list. Treat "
        "these six lists as six separate documents: never cite or "
        "reference one dimension's inputs when scoring or summarizing a "
        "different dimension, even if it seems relevant.\n\n" + _RUBRIC
    )

    forward_guidance_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Forward Guidance, formatted as '[<index>] <text>' one per line"
    )
    uncertainty_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Uncertainty, formatted as '[<index>] <text>' one per line"
    )
    confidence_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Confidence, formatted as '[<index>] <text>' one per line"
    )
    sentiment_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Sentiment, formatted as '[<index>] <text>' one per line"
    )
    macro_focus_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Macro Focus, formatted as '[<index>] <text>' one per line"
    )
    jargon_inputs: list[str] = dspy.InputField(
        desc="Indexed composites for Jargon, formatted as '[<index>] <text>' one per line"
    )
    is_terminal: bool = dspy.InputField(
        desc="True if this group's composites are a final conclusion, not input to a further round"
    )

    forward_guidance_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Forward-Guidance input, IN ORDER — length must equal len(forward_guidance_inputs)"
    )
    forward_guidance_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    forward_guidance_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    uncertainty_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Uncertainty input, IN ORDER — length must equal len(uncertainty_inputs)"
    )
    uncertainty_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    uncertainty_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    confidence_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Confidence input, IN ORDER — length must equal len(confidence_inputs)"
    )
    confidence_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    confidence_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    sentiment_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Sentiment input, IN ORDER — length must equal len(sentiment_inputs)"
    )
    sentiment_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    sentiment_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    macro_focus_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Macro-Focus input, IN ORDER — length must equal len(macro_focus_inputs)"
    )
    macro_focus_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    macro_focus_label: int | None = dspy.OutputField(desc=_LABEL_DESC)

    jargon_weights: list[int] = dspy.OutputField(
        desc="One integer 0-3 per Jargon input, IN ORDER — length must equal len(jargon_inputs)"
    )
    jargon_summary: str = dspy.OutputField(desc=_SUMMARY_DESC)
    jargon_label: int | None = dspy.OutputField(desc=_LABEL_DESC)
