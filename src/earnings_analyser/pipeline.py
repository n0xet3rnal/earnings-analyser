from dataclasses import dataclass

import dspy

from .data.splitter import SplitConfidence
from .modules.section_scorer import SectionScorer, SectionScores
from .modules.section_splitter import SectionSplitter

DEFAULT_N_SAMPLES = 3


@dataclass(frozen=True)
class TranscriptScores:
    ticker: str
    earnings_date: str
    transcript_text: str  # the full original document — evidence spans map back into this
    split_confidence: SplitConfidence
    prepared_remarks: SectionScores
    prepared_remarks_offset: int  # where prepared_remarks starts within transcript_text
    qa: SectionScores | None  # None if the transcript had no detectable Q&A section
    qa_offset: int | None  # where qa_section starts within transcript_text


class TranscriptScorer(dspy.Module):
    """End-to-end: split a raw transcript into prepared remarks / Q&A
    (using the model's own understanding of the call's structure, not
    regex — see modules.section_splitter), then score both sections
    across all 6 dimensions."""

    def __init__(self, n_samples: int = DEFAULT_N_SAMPLES):
        super().__init__()
        self.splitter = SectionSplitter()
        self.section_scorer = SectionScorer(n_samples=n_samples)

    def forward(self, ticker: str, earnings_date: str, transcript: str) -> TranscriptScores:
        split = self.splitter(transcript_text=transcript)

        prepared = self.section_scorer(
            section_text=split.prepared_remarks, section_type="prepared_remarks"
        )
        qa = (
            self.section_scorer(section_text=split.qa_section, section_type="qa")
            if split.qa_section
            else None
        )

        # Both sections are exact (stripped) substrings of the original
        # transcript by construction — locate them so evidence spans
        # (currently relative to their own section text) can be mapped
        # back into the full document for highlighting.
        prepared_offset = transcript.find(split.prepared_remarks) if split.prepared_remarks else 0
        qa_offset = transcript.find(split.qa_section) if split.qa_section else None

        return TranscriptScores(
            ticker=ticker,
            earnings_date=earnings_date,
            transcript_text=transcript,
            split_confidence=split.confidence,
            prepared_remarks=prepared,
            prepared_remarks_offset=max(prepared_offset, 0),
            qa=qa,
            qa_offset=qa_offset if qa_offset is not None and qa_offset >= 0 else None,
        )
