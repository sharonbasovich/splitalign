from splitalign.align import AlignOp
from splitalign.judge import TagJudgment
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


def test_score_tag_flagged_tokens():
    ta, sa = _doc("One two three .")
    tb, sb = _doc("Uno dos tres .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9)]
    j = TagJudgment(a_ids=[0], b_ids=[1, 2])
    la, lb, st = score_item(ta, tb, sa, sb, "One two three .",
                            "Uno dos tres .", ops, [j])
    assert la == [1.0, 0.0, 0.0, -1.0]
    assert lb == [0.0, 1.0, 1.0, -1.0]
    assert st.flagged_a == 1 and st.flagged_b == 2
    assert st.valid_matched_tokens == 8 and st.matched_tokens == 8
    assert st.invalid_pairs == 0


def test_score_empty_tags_all_zero():
    ta, sa = _doc("One two .")
    tb, sb = _doc("Uno dos .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9)]
    j = TagJudgment(a_ids=[], b_ids=[])
    la, lb, st = score_item(ta, tb, sa, sb, "One two .", "Uno dos .", ops, [j])
    assert la == [0.0, 0.0, -1.0]
    assert lb == [0.0, 0.0, -1.0]
    assert st.flagged_a == 0 and st.flagged_b == 0


def test_score_omit_add():
    ta, sa = _doc("A B .")
    tb, sb = _doc("X .")
    ops = [AlignOp("1:0", 0, 1, -1, -1, 0.0), AlignOp("0:1", -1, -1, 0, 1, 0.0)]
    la, lb, st = score_item(ta, tb, sa, sb, "A B .", "X .", ops, [None, None],
                            ScoreConfig(omit_label=1.0, add_label=0.8))
    assert la[0] == 1.0 and la[1] == 1.0 and la[2] == -1.0
    assert lb[0] == 0.8 and lb[1] == -1.0


def test_punct_ids_dropped_and_counted():
    ta, sa = _doc("a , b .")
    tb, sb = _doc("c d .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9)]
    # "," is index 1 and "." index 3 in a — flagged punct ids must drop, count
    j = TagJudgment(a_ids=[0, 1, 3], b_ids=[2])
    la, lb, st = score_item(ta, tb, sa, sb, "a , b .", "c d .", ops, [j])
    assert la == [1.0, -1.0, 0.0, -1.0]
    assert lb == [0.0, 0.0, -1.0]
    assert st.dropped_punct_ids == 3
    assert st.flagged_a == 1


def test_invalid_judgment_score0_fallback():
    ta, sa = _doc("a b .")
    tb, sb = _doc("c d .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.2)]
    j = TagJudgment(a_ids=[], b_ids=[], ok=False, repairs=1)
    la, lb, st = score_item(ta, tb, sa, sb, "a b .", "c d .", ops, [j])
    assert st.parse_failures == 1 and st.invalid_pairs == 1
    assert st.invalid_fallback_tokens == 6
    assert la[:2] == [0.0, 0.0] and lb[:2] == [0.0, 0.0]
    assert st.valid_matched_tokens == 0 and st.matched_tokens == 6


def test_judge_valid_token_coverage_mix():
    ta, sa = _doc("a b . x y .")
    tb, sb = _doc("c d . u v .")
    ops = [AlignOp("1:1", 0, 1, 0, 1, 0.9), AlignOp("1:1", 1, 2, 1, 2, 0.9)]
    js = [TagJudgment(a_ids=[], b_ids=[0]),
          TagJudgment(a_ids=[], b_ids=[], ok=False)]
    la, lb, st = score_item(ta, tb, sa, sb, "a b . x y .", "c d . u v .",
                            ops, js)
    # unique-token accounting: 6 valid, 6 under declared fallback
    assert st.valid_matched_tokens == 6 and st.matched_tokens == 12
    assert st.invalid_pairs == 1 and st.invalid_fallback_tokens == 6
