"""Monotone dynamic-programming sentence alignment.

Operations (per the SplitAlign design):

* ``1:1`` — one source sentence ↔ one target sentence
* ``1:0`` — source sentence omitted on the target side
* ``0:1`` — target sentence added (no source counterpart)
* ``1:2`` — one source sentence split across two target sentences

Costs come from a similarity matrix sim[i][j] in [0, 1]. Deletion/insertion
penalties and the boilerplate guard are calibratable. Ties resolve
deterministically: 1:1 > 1:2 > 1:0 > 0:1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

OP_MATCH = "1:1"
OP_SPLIT = "1:2"
OP_OMIT = "1:0"
OP_ADD = "0:1"


@dataclass(frozen=True)
class AlignOp:
    op: str
    a_start: int
    a_end: int    # exclusive
    b_start: int
    b_end: int    # exclusive
    sim: float


@dataclass
class AlignConfig:
    omit_cost: float = 0.55        # penalty for 1:0
    add_cost: float = 0.55         # penalty for 0:1
    split_bias: float = 0.20       # extra cost for 1:2 vs match+add
    boilerplate_floor: float = 0.15  # below this sim, matching is discouraged
    boilerplate_penalty: float = 0.35


def align(sim: list[list[float]], cfg: AlignConfig | None = None) -> list[AlignOp]:
    """Align n x m similarity matrix into a monotone op sequence."""
    cfg = cfg or AlignConfig()
    n = len(sim)
    m = len(sim[0]) if n else 0
    if n == 0 or m == 0:
        ops = [AlignOp(OP_OMIT, i, i + 1, -1, -1, 0.0) for i in range(n)]
        ops += [AlignOp(OP_ADD, -1, -1, j, j + 1, 0.0) for j in range(m)]
        return ops

    NEG = float("inf")
    D = [[NEG] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            cands: list[tuple[float, tuple]] = []
            if i and j:
                s = sim[i - 1][j - 1]
                c = (1.0 - s) + (cfg.boilerplate_penalty if s < cfg.boilerplate_floor else 0.0)
                cands.append((D[i - 1][j - 1] + c, (i - 1, j - 1, OP_MATCH, s)))
            if i and j >= 2:
                s12 = (sim[i - 1][j - 2] + sim[i - 1][j - 1]) / 2.0
                c = (1.0 - s12) + cfg.split_bias
                cands.append((D[i - 1][j - 2] + c, (i - 1, j - 2, OP_SPLIT, s12)))
            if i:
                cands.append((D[i - 1][j] + cfg.omit_cost, (i - 1, j, OP_OMIT, 0.0)))
            if j:
                cands.append((D[i][j - 1] + cfg.add_cost, (i, j - 1, OP_ADD, 0.0)))
            # deterministic tie-break: order above defines preference
            best = min(cands, key=lambda c: (c[0], [OP_MATCH, OP_SPLIT, OP_OMIT, OP_ADD].index(c[1][2])))
            D[i][j], back[i][j] = best

    ops: list[AlignOp] = []
    i, j = n, m
    while i or j:
        pi, pj, op, s = back[i][j]
        if op == OP_MATCH:
            ops.append(AlignOp(op, pi, pi + 1, pj, pj + 1, s))
        elif op == OP_SPLIT:
            ops.append(AlignOp(op, pi, pi + 1, pj, pj + 2, s))
        elif op == OP_OMIT:
            ops.append(AlignOp(op, pi, pi + 1, -1, -1, 0.0))
        else:
            ops.append(AlignOp(op, -1, -1, pj, pj + 1, 0.0))
        i, j = pi, pj
    ops.reverse()
    return ops


def coverage(ops: list[AlignOp], n: int, m: int) -> dict:
    """Sanity stats: every sentence covered exactly once, order preserved."""
    a_used: list[int] = []
    b_used: list[int] = []
    for op in ops:
        a_used += list(range(op.a_start, op.a_end)) if op.a_start >= 0 else []
        b_used += list(range(op.b_start, op.b_end)) if op.b_start >= 0 else []
    return {
        "a_complete": a_used == list(range(n)),
        "b_complete": b_used == list(range(m)),
        "n_ops": len(ops),
    }
