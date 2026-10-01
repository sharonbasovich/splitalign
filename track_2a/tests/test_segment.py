from splitalign.segment import (is_punctuation, segment_sentences,
                                segment_text, tokens_of)


def test_tokens_preserve_offsets():
    text = "Hello  world , again ."
    toks = tokens_of(text)
    assert [t.text for t in toks] == ["Hello", "world", ",", "again", "."]
    assert toks[0].start == 0 and toks[0].end == 5
    assert text[toks[3].start:toks[3].end] == "again"
    # every token's offsets slice back its own text, even with doublespaces
    for t in toks:
        assert text[t.start:t.end] == t.text


def test_punctuation_detection():
    assert is_punctuation(".")
    assert is_punctuation("—")
    assert is_punctuation("(?)")
    assert not is_punctuation("word")
    assert not is_punctuation("6.8")
    assert not is_punctuation("end.")


def test_sentence_split():
    toks, segs = segment_text("Energy policy . Switzerland imports fuel .")
    assert len(segs) == 2
    a = toks[segs[0].start_token:segs[0].end_token]
    assert [t.text for t in a] == ["Energy", "policy", "."]


def test_heading_and_abbrev():
    toks, segs = segment_text("e.g. something ended . Next starts now .")
    assert len(segs) == 2  # no split after e.g.


def test_empty():
    assert segment_sentences([]) == []


def test_end_of_doc_no_terminal():
    toks, segs = segment_text("One sentence . trailing fragment without period")
    assert segs[-1].end_token == len(toks)


def test_newline_paragraphs():
    toks, segs = segment_text("First block .\n\nSecond block .")
    assert len(segs) == 2
