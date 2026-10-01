from splitalign.align import (OP_ADD, OP_MATCH, OP_OMIT, OP_SPLIT, align,
                              coverage)


def test_perfect_alignment():
    sim = [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]
    ops = align(sim)
    assert [o.op for o in ops] == [OP_MATCH] * 4
    assert coverage(ops, 4, 4)["a_complete"]


def test_trailing_omission_and_addition():
    # a has 3 sentences, b has 5: extra a -> omit, extra b -> add
    sim = [[0.9, 0.0, 0.0, 0.0, 0.0],
           [0.0, 0.9, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.9, 0.0, 0.0]]
    ops = align(sim)
    assert ops[-1].op == OP_ADD and ops[-2].op == OP_ADD
    assert ops[0].op == OP_MATCH


def test_split_op():
    sim = [[0.8, 0.8]]
    ops = align(sim)
    assert ops[0].op == OP_SPLIT


def test_empty_side():
    from splitalign.align import AlignOp
    ops = align([[]])  # 1 source sentence, 0 target sentences
    assert ops == [AlignOp(OP_OMIT, 0, 1, -1, -1, 0.0)]


def test_monotone_coverage():
    import random
    random.seed(0)
    n, m = 12, 15
    sim = [[random.random() for _ in range(m)] for _ in range(n)]
    ops = align(sim)
    cov = coverage(ops, n, m)
    assert cov["a_complete"] and cov["b_complete"]
