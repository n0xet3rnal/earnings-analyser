from earnings_analyser.data.sentence_split import base_windows, split_sentences


def test_offsets_round_trip_to_source_text():
    text = "Revenue grew 12%. Margins held steady. We remain cautious about next quarter."
    sentences = split_sentences(text)

    assert [s.text for s in sentences] == [
        "Revenue grew 12%.",
        "Margins held steady.",
        "We remain cautious about next quarter.",
    ]
    for s in sentences:
        assert text[s.start : s.end] == s.text


def test_offsets_survive_irregular_whitespace():
    text = "First sentence.   Second sentence.\n\nThird one here."
    sentences = split_sentences(text)
    for s in sentences:
        assert text[s.start : s.end] == s.text


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_base_windows_overlap_by_one_sentence():
    text = " ".join(f"Sentence number {i}." for i in range(12))
    sentences = split_sentences(text)
    windows = base_windows(sentences, size=5, overlap=1)

    for prev, nxt in zip(windows, windows[1:]):
        assert prev.sentences[-1].index == nxt.sentences[0].index

    # every sentence appears in at least one window
    covered = {s.index for w in windows for s in w.sentences}
    assert covered == {s.index for s in sentences}


def test_base_windows_rejects_overlap_ge_size():
    import pytest

    with pytest.raises(ValueError):
        base_windows([], size=4, overlap=4)


def test_base_windows_handles_short_transcript():
    text = "Only one sentence here."
    sentences = split_sentences(text)
    windows = base_windows(sentences, size=5, overlap=1)
    assert len(windows) == 1
    assert [s.text for s in windows[0].sentences] == ["Only one sentence here."]
