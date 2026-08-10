"""Load and lightly normalize the glopardo/sp500-earnings-transcripts dataset."""

from dataclasses import dataclass

from datasets import Dataset, load_dataset

DATASET_NAME = "glopardo/sp500-earnings-transcripts"


@dataclass(frozen=True)
class TranscriptRecord:
    ticker: str
    company: str
    sector: str
    earnings_date: str
    year: float
    quarter: float
    eps12mtrailing_eoq: float | None
    eps12mfwd_qavg: float | None
    transcript: str


def load_transcripts() -> Dataset:
    """Return the raw HF dataset (single 'train' split, ~20.7k rows)."""
    return load_dataset(DATASET_NAME)["train"]


def to_record(row: dict) -> TranscriptRecord:
    return TranscriptRecord(
        ticker=row["ticker"],
        company=row["company"],
        sector=row["sector"],
        earnings_date=row["earnings_date"],
        year=row["year"],
        quarter=row["quarter"],
        eps12mtrailing_eoq=row["eps12mtrailing_eoq"],
        eps12mfwd_qavg=row["eps12mfwd_qavg"],
        transcript=row["transcript"],
    )


def forward_eps_revision(dataset: Dataset, ticker: str, earnings_date: str) -> float | None:
    """Option A metric input: % change in forward-12m EPS estimate vs. the
    prior quarter's call for the same ticker.

    Positive = analysts got more bullish after this call (revision-up proxy
    for a positive surprise). Returns None if there's no prior quarter or
    either estimate is missing/zero.
    """
    df = dataset.to_pandas()
    ticker_rows = df[df["ticker"] == ticker].sort_values("earnings_date").reset_index(drop=True)

    match_idx = ticker_rows.index[ticker_rows["earnings_date"] == earnings_date]
    if len(match_idx) == 0 or match_idx[0] == 0:
        return None
    i = match_idx[0]

    curr = ticker_rows.loc[i, "eps12mfwd_qavg"]
    prev = ticker_rows.loc[i - 1, "eps12mfwd_qavg"]
    if curr is None or prev in (None, 0) or (curr != curr) or (prev != prev):  # NaN check
        return None
    return (curr - prev) / abs(prev)
