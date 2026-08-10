"""Regex-based transcript splitter — FALLBACK ONLY.

This was tuned against glopardo/sp500-earnings-transcripts specifically
(98.4% clean-split rate on that dataset's phrasing conventions: "first
question" as the operator hand-off cue, "[Operator Instructions]" as a
secondary cue). It does NOT generalize to arbitrary transcript sources
(other vendors, company-published PDFs, OCR'd scans) with different
phrasing — that's what `modules.section_splitter.SectionSplitter` (LLM-
based, reads the call's actual structure) is for. This module exists as
its cheap fallback when that LLM call fails outright.
"""

import re
from dataclasses import dataclass
from typing import Literal

_FIRST_QUESTION = re.compile(r"\bfirst question\b", re.IGNORECASE)
_OPERATOR_INSTRUCTIONS = re.compile(r"[\[(]operator instructions?\.?[\])]", re.IGNORECASE)

SplitConfidence = Literal["llm_verified", "llm_no_qa", "first_question", "operator_instructions", "none"]


@dataclass(frozen=True)
class SplitResult:
    prepared_remarks: str
    qa_section: str
    confidence: SplitConfidence


def regex_split_transcript(text: str) -> SplitResult:
    for pattern, confidence in (
        (_FIRST_QUESTION, "first_question"),
        (_OPERATOR_INSTRUCTIONS, "operator_instructions"),
    ):
        match = pattern.search(text)
        if match:
            # back up to the start of the sentence/turn so we don't cut mid-clause
            cut = text.rfind(".", 0, match.start())
            cut = cut + 1 if cut != -1 else match.start()
            return SplitResult(
                prepared_remarks=text[:cut].strip(),
                qa_section=text[cut:].strip(),
                confidence=confidence,
            )

    return SplitResult(prepared_remarks=text.strip(), qa_section="", confidence="none")
