"""Splits a transcript into prepared remarks vs. Q&A using the model's own
understanding of the call's structure — not regex pattern matching tuned
to one dataset's phrasing. This is what makes the pipeline work on
arbitrary raw input (pasted text, uploaded PDFs), not just the curated
HuggingFace dataset.

The model returns a short verbatim marker for where Q&A begins; we locate
that marker in the actual source text with the same verified-substring
technique used for evidence quotes (never trust the model's own
reproduction of text — confirm it, then split at that exact offset in
code). Retries a few times (temperature=0.3, cache disabled) before
giving up, because — as with DimensionScores (see
scripts/test_signatures.py) — Gemini does not reliably reproduce the
same marker for the same input even at temperature=0, so a single
unverified attempt is not a reliable "there's no Q&A section" signal.
Falls back to the cheap regex splitter only if every attempt fails to
produce a verifiable marker.
"""

import os

import dspy

from ..config import DEFAULT_MODEL
from ..data.splitter import SplitResult, regex_split_transcript
from ..signatures import SectionBoundary
from ..text_matching import find_marker

DEFAULT_MAX_ATTEMPTS = 3
RETRY_TEMPERATURE = 0.3


class SectionSplitter(dspy.Module):
    def __init__(self, max_attempts: int = DEFAULT_MAX_ATTEMPTS):
        super().__init__()
        self.max_attempts = max_attempts
        self.predict = dspy.Predict(SectionBoundary)
        self._retry_lm = dspy.LM(
            DEFAULT_MODEL,
            api_key=os.environ["GEMINI_API_KEY"],
            temperature=RETRY_TEMPERATURE,
            cache=False,
        )

    def forward(self, transcript_text: str) -> SplitResult:
        saw_no_qa = False

        for _ in range(self.max_attempts):
            try:
                with dspy.context(lm=self._retry_lm):
                    result = self.predict(transcript_text=transcript_text)
            except Exception:
                continue

            if not result.has_qa_section or not result.qa_start_marker.strip():
                saw_no_qa = True
                continue

            idx = find_marker(result.qa_start_marker, transcript_text)
            if idx is not None:
                return SplitResult(
                    prepared_remarks=transcript_text[:idx].strip(),
                    qa_section=transcript_text[idx:].strip(),
                    confidence="llm_verified",
                )
            # Claimed a marker but it doesn't verify — try again rather
            # than trusting it or giving up immediately.

        if saw_no_qa:
            return SplitResult(
                prepared_remarks=transcript_text.strip(), qa_section="", confidence="llm_no_qa"
            )

        return regex_split_transcript(transcript_text)
