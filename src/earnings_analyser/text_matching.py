"""Verified substring matching: locate model-produced text spans in a
source document without trusting the model's own reproduction of them.

Used for evidence-quote verification (does this citation really appear
in the transcript, and where — for highlighting it in place), and for
section-boundary detection (does the claimed Q&A start marker really
appear, and where?). Never assume a model-returned "verbatim" excerpt is
actually verbatim — check it.
"""

import re


def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).strip()


def locate_span(marker: str, source_text: str) -> tuple[int, int] | None:
    """Return the (start, end) character offsets of `marker` in
    `source_text` — exact match preferred, falling back to
    whitespace/smart-quote-tolerant matching (tokenize on whitespace,
    join with \\s+, treat straight and smart quotes as equivalent) — or
    None if it genuinely isn't there."""
    marker = marker.strip()
    if not marker:
        return None

    idx = source_text.find(marker)
    if idx != -1:
        return idx, idx + len(marker)

    tokens = marker.split()
    if not tokens:
        return None

    def token_pattern(tok: str) -> str:
        pattern = re.escape(tok)
        pattern = pattern.replace(re.escape("'"), "['‘’]")
        pattern = pattern.replace(re.escape('"'), '["“”]')
        return pattern

    pattern = r"\s+".join(token_pattern(tok) for tok in tokens)
    match = re.search(pattern, source_text)
    return (match.start(), match.end()) if match else None


def find_marker(marker: str, source_text: str) -> int | None:
    """Return just the start offset — see `locate_span`."""
    span = locate_span(marker, source_text)
    return span[0] if span else None


def verify_quote(quote: str, source_text: str) -> bool:
    """True if `quote` appears verbatim (modulo whitespace/smart-quote
    normalization) in `source_text`."""
    if not quote.strip():
        return False
    return normalize_text(quote) in normalize_text(source_text)
