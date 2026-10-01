from splitalign.align import AlignOp
from splitalign.judge import Judgment
from splitalign.score import ScoreConfig, map_span_to_token_range, score_item
from splitalign.segment import segment_text


def _doc(text):
    toks, segs = segment_text(text)
    return toks, segs


def test_span_mapping_repeated_tokens():
    text = "the cat sat on the mat"
    toks, _ = _doc(text)
    # "the" appears twice; span "the mat" must land at indices 4..6
    rng = map_span_to_token_range(text, toks, "the mat")
    assert rng == (4, 6)
    rng = map_span_to_token_range(text, toks, "the", search_from=toks[4].start)
    assert rng == (4, 5)


def test_score_basic_ops():
    ta, sa = _doc("One two .")
    tb, sb = _doc("Uno dos .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9)]
    j = Judgment(difference=1, spans_a=[], spans_b=[])
    la, lb, st = score_item(ta, tb, sa, sb, "One two .", "Uno dos .", ops, [j])
    assert la == [0.2, 0.2, -1.0]
    assert lb == [0.2, 0.2, -1.0]


def test_score_omit_add():
    ta, sa = _doc("A B .")
    tb, sb = _doc("X .")
    ops = [AlignOp("1:0", 0, 1, -1, -1, 0.0), AlignOp("0:1", -1, -1, 0, 1, 0.0)]
    la, lb, st = score_item(ta, tb, sa, sb, "A B .", "X .", ops, [None, None],
                            ScoreConfig(omit_label=1.0, add_label=0.8))
    assert la[0] == 1.0 and la[1] == 1.0 and la[2] == -1.0
    assert lb[0] == 0.8 and lb[1] == -1.0


def test_score_span_boost():
    text_a = "alpha beta gamma ."
    text_b = "alpha beta delta ."
    ta, sa = _doc(text_a)
    tb, sb = _doc(text_b)
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9)]
    j = Judgment(difference=2, spans_a=["gamma"], spans_b=["delta"])
    la, lb, st = score_item(ta, tb, sa, sb, text_a, text_b, ops, [j],
                            ScoreConfig(span_boost=0.3, span_floor=0.6))
    assert la[0] == la[1] == 0.4  # base = 2/5
    assert la[2] == 0.7           # boosted
    assert lb[2] == 0.7
    assert st.span_matched == 2


def test_failed_judgment_falls_back():
    ta, sa = _doc("a b .")
    tb, sb = _doc("c d .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.2)]
    j = Judgment(difference=-1, spans_a=[], spans_b=[], ok=False)
    la, lb, st = score_item(ta, tb, sa, sb, "a b .", "c d .", ops, [j])
    assert st.parse_failures == 1
    assert abs(la[0] - 0.8) < 1e-6  # (1-sim)*scale
