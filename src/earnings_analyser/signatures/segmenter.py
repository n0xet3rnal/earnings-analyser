import dspy


class Segmenter(dspy.Signature):
    """Group the indexed items below into at most `max_clusters` clusters by
    which items belong to the same underlying thread — connected by shared
    topic, causal explanation, rhetorical setup-and-payoff, contrastive
    rebuttal, or reference back to something said earlier. Items in the same
    thread do not need matching vocabulary. Then flag which items in each
    cluster you actually used best represent it on their own.

    cluster_id: one integer per item, IN ORDER, 0 to max_clusters-1.
    is_representative: one integer per item, IN ORDER, 1 if flagged, else 0.
    Integers only, exactly one entry per item, in order — never a word,
    never a mapping, never fewer or more entries than there are items."""

    items: list[str] = dspy.InputField(desc="Indexed items, '[<index>] <text>' one per line")
    max_clusters: int = dspy.InputField()

    cluster_id: list[int] = dspy.OutputField(desc="One int 0..max_clusters-1 per item, IN ORDER")
    is_representative: list[int] = dspy.OutputField(desc="One int 0/1 per item, IN ORDER")
