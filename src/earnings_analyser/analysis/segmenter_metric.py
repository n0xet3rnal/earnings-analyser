"""Pure-code metric for the segmenter optimization (implementation plan:
DSPy-optimize a segmenter). No judge model — ground truth is known by
construction from the synthetic data generator.
"""

import itertools
from collections import Counter


def pairwise_agreement(predicted: list[int], truth: list[int]) -> float:
    """Fraction of item pairs where predicted same/different-cluster
    agrees with ground-truth same/different-thread.

    NOT chance-corrected — a degenerate "every item its own cluster"
    prediction scores ~0.68 on real held-out data (measured directly),
    because most pairs in a multi-cluster ground truth are cross-cluster
    anyway, and "predict everything different" gets those right for
    free. This makes the raw score unable to distinguish real clustering
    from total failure. Kept for the pair-level detail; use
    `adjusted_rand_index` as the actual optimization/eval metric.
    """
    n = len(truth)
    predicted = (list(predicted) + [-1] * n)[:n]
    pairs = list(itertools.combinations(range(n), 2))
    if not pairs:
        return 1.0
    agree = sum((predicted[i] == predicted[j]) == (truth[i] == truth[j]) for i, j in pairs)
    return agree / len(pairs)


def _n_choose_2(x: int) -> int:
    return x * (x - 1) // 2


def adjusted_rand_index(predicted: list[int], truth: list[int]) -> float:
    """Chance-corrected clustering agreement: 1.0 = perfect match (up to
    relabeling), ~0.0 = no better than a random/degenerate partition,
    negative = worse than chance. Fixes pairwise_agreement's bias toward
    over-fragmented (or otherwise degenerate) predictions.

    Standard contingency-table ARI (Hubert & Arabie 1985). No sklearn
    dependency — this is ~15 lines, not worth a new dependency for.
    """
    n = len(truth)
    predicted = (list(predicted) + [-1] * n)[:n]
    if n == 0:
        return 1.0

    contingency: Counter[tuple[int, int]] = Counter(zip(predicted, truth))
    row_sums = Counter(predicted)
    col_sums = Counter(truth)

    index = sum(_n_choose_2(c) for c in contingency.values())
    expected = sum(_n_choose_2(a) for a in row_sums.values()) * sum(_n_choose_2(b) for b in col_sums.values())
    expected /= _n_choose_2(n) if n >= 2 else 1
    max_index = 0.5 * (sum(_n_choose_2(a) for a in row_sums.values()) + sum(_n_choose_2(b) for b in col_sums.values()))

    denom = max_index - expected
    if denom == 0:
        return 1.0 if index == max_index else 0.0
    return (index - expected) / denom
