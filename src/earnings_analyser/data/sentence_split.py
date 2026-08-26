"""Deterministic sentence splitting and base-window grouping.

No model involved anywhere in this module. Every sentence carries a real,
absolute offset into the source transcript, computed once here — this is
the leaf layer of the collapse graph (see `modules/collapse_step.py`), and
nothing above this layer ever reproduces or hallucinates source text
because nothing above it *is* source text.

Base windows can overlap by N sentences (each window's last N sentences
also open the next window) — this was the only source of multi-parent
structure in the collapse graph (a leaf sentence at a boundary weighted
by two separate calls, cited by two composites). The pipeline's default
is now 0 overlap: call-count reduction (implementation-plan.md, "Cut
collapse-pipeline wall-clock time") took priority over that boundary
guarantee, so the live collapse graph is currently a strict tree
end-to-end, not a DAG. `overlap` stays a supported parameter here — set
it above 0 to restore the old guarantee if boundary-split evidence turns
out to matter more than call count.
"""

import re
from dataclasses import dataclass

DEFAULT_WINDOW_SIZE = 5
DEFAULT_WINDOW_OVERLAP = 1

# Split after sentence-ending punctuation followed by whitespace and a
# capital letter, digit, or quote — deliberately simple and deterministic
# rather than a full NLP sentence tokenizer. Good enough for earnings-call
# transcripts; not aiming for perfect boundary detection on abbreviations.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


@dataclass(frozen=True)
class Sentence:
    index: int
    start: int  # absolute offset into the source transcript
    end: int
    text: str


@dataclass(frozen=True)
class Window:
    index: int
    sentences: tuple[Sentence, ...]


def split_sentences(text: str) -> list[Sentence]:
    """Deterministically split `text` into sentences with real offsets."""
    sentences: list[Sentence] = []
    cursor = 0

    def _emit(chunk_start: int, chunk: str) -> None:
        stripped = chunk.strip()
        if not stripped:
            return
        lead = len(chunk) - len(chunk.lstrip())
        real_start = chunk_start + lead
        sentences.append(Sentence(len(sentences), real_start, real_start + len(stripped), stripped))

    for match in _SENTENCE_END.finditer(text):
        _emit(cursor, text[cursor:match.start()])
        cursor = match.end()
    _emit(cursor, text[cursor:])

    return sentences


def base_windows(
    sentences: list[Sentence],
    size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_WINDOW_OVERLAP,
) -> list[Window]:
    """Group sentences into overlapping windows of `size` sentences,
    advancing by `size - overlap` each step so consecutive windows share
    `overlap` sentences at the boundary."""
    if overlap >= size:
        raise ValueError("window overlap must be smaller than window size")
    if not sentences:
        return []

    step = size - overlap
    windows: list[Window] = []
    i = 0
    n = len(sentences)
    while i < n:
        chunk = tuple(sentences[i : i + size])
        windows.append(Window(len(windows), chunk))
        if i + size >= n:
            break
        i += step
    return windows
