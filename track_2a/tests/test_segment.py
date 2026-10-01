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
    """Official -1 convention: single ASCII non-alnum char only."""
    for p in ". , ; : ! ? - / % + | & ' \" @ * ( ) < > [ ] _":
        assert is_punctuation(p), p
    assert not is_punctuation("—")   # em-dash: scored, not -1 in gold
    assert not is_punctuation("–")   # en-dash
    assert not is_punctuation("«")   # guillemet
    assert not is_punctuation("©")
    assert not is_punctuation("€")
    assert not is_punctuation("…")
    assert not is_punctuation("(?)")  # multi-char tokens are never -1
    assert not is_punctuation("word")
    assert not is_punctuation("6.8")
    assert not is_punctuation("end.")


def test_punctuation_parity_with_dev_gold():
    """is_punctuation must match gold -1 labels on every dev token."""
    import glob
    import json
    mismatches = []
    for f in glob.glob("data/gold/dev/*/gold_*.jsonl"):
        for line in open(f):
            g = json.loads(line)
            for side in ("a", "b"):
                for tok, lab in zip(g[f"text_{side}"].split(), g[f"labels_{side}"]):
                    if is_punctuation(tok) != (lab == -1):
                        mismatches.append((f, tok, lab))
    assert not mismatches, mismatches[:10]


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
