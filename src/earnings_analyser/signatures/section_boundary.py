import dspy


class SectionBoundary(dspy.Signature):
    """Identify where the prepared remarks end and the Q&A begins in this
    earnings call transcript. Formats vary by source — some have an explicit
    operator hand-off ("first question comes from..."), some don't use that
    phrasing at all, some have no Q&A section (e.g. a presentation-only
    call). Use your understanding of the call's structure, not a fixed
    phrase — look for where the tone shifts from a scripted monologue to a
    back-and-forth exchange with named analysts.

    If a Q&A section exists, return has_qa_section=True and a short
    (10-25 word) marker copied VERBATIM, word-for-word, from the transcript
    — starting at the first word of the sentence/turn that begins the Q&A
    portion (the operator's hand-off line if there is one, otherwise the
    first analyst's question). Do not paraphrase or fix punctuation.

    If there is no Q&A section at all, return has_qa_section=False and an
    empty marker.
    """

    transcript_text: str = dspy.InputField()
    has_qa_section: bool = dspy.OutputField()
    qa_start_marker: str = dspy.OutputField()
