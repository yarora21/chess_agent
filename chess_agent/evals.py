"""Win% conversion and move classification.

This module is the whole of "what counts as a blunder". Nothing else in the
project is allowed to decide that, and the LLM narrator never sees a position --
it only ever reads the labels produced here.

Scores are centipawns from White's point of view throughout, matching Lichess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Lichess's win-probability curve. Fixed constant, published as part of their
# accuracy metric: Win% = 50 + 50 * (2 / (1 + exp(-0.00368208 * cp)) - 1).
WIN_PCT_K = 0.00368208

# Classification thresholds, in win% lost by the mover on a single move.
#
# These are NOT invented: they were recovered from Lichess's own judgments on
# the 307 server-analysed games in this store (21,157 judged plies). Fitting a
# threshold rule against their "Blunder" label puts the cut at 15 win% -- at
# that value we agree on 1,661 of the 1,664 moves we flag, i.e. 3 false
# positives. The upper edges of their Inaccuracy and Mistake classes land on
# 10.00 and 14.99 win% respectively, which fixes the two lower cuts.
#
# Reproducing Lichess's classifier is the point: phase 2's gate compares our
# blunder counts to theirs, and that comparison is only meaningful if we are
# running the same rule over our own evals.
BLUNDER_WIN_PCT = 15.0
MISTAKE_WIN_PCT = 10.0
INACCURACY_WIN_PCT = 5.0

# Moves played from an already-decided position are excluded from ACPL-style
# metrics: swings there are noise, not skill. Capped ACPL ignores them anyway.
DECIDED_CP = 800


def win_pct(cp: int | None = None, mate: int | None = None) -> float:
    """Win probability for White, 0-100.

    Exactly one of cp/mate is expected, matching the `evals` table's CHECK.
    """
    if mate is not None:
        return 100.0 if mate > 0 else 0.0
    if cp is None:
        raise ValueError("win_pct needs one of cp / mate")
    return 50 + 50 * (2 / (1 + math.exp(-WIN_PCT_K * cp)) - 1)


def win_pct_lost(before: float, after: float, mover_is_white: bool) -> float:
    """Win probability the mover gave up. Positive means the move cost something.

    `before` and `after` are both White-POV win percentages; flipping the sign
    for Black is the single place where point of view is handled, so a sign bug
    has exactly one place to hide.
    """
    return (before - after) if mover_is_white else (after - before)


@dataclass(frozen=True)
class Judgment:
    name: str          # 'blunder' | 'mistake' | 'inaccuracy' | 'ok'
    win_pct_lost: float

    @property
    def is_blunder(self) -> bool:
        return self.name == "blunder"


def classify(win_pct_lost_value: float) -> Judgment:
    if win_pct_lost_value >= BLUNDER_WIN_PCT:
        name = "blunder"
    elif win_pct_lost_value >= MISTAKE_WIN_PCT:
        name = "mistake"
    elif win_pct_lost_value >= INACCURACY_WIN_PCT:
        name = "inaccuracy"
    else:
        name = "ok"
    return Judgment(name, win_pct_lost_value)


@dataclass(frozen=True)
class MoveEval:
    """Everything the metrics layer needs about one move, derived from two evals."""

    ply: int
    mover_is_white: bool
    win_pct_before: float
    win_pct_after: float
    judgment: Judgment
    cp_loss: float | None   # centipawn loss from the mover's POV; None if mate involved
    decided: bool           # position was already decided before the move

    @property
    def is_blunder(self) -> bool:
        return self.judgment.is_blunder


def judge_game(rows: "list", initial_cp: int | None) -> dict[int, MoveEval]:
    """Judge every ply of one game from its eval rows.

    `rows` hold the eval of the position AFTER each ply, so the 'before' side of
    ply i is row i-1 -- and for ply 0 it is the engine's eval of the starting
    position, which is why engine_configs.initial_cp exists.

    Shared by the metrics layer and the Lichess-agreement gate so the two can
    never drift apart: if this is wrong, validate.py fails loudly.
    """
    by_ply = {r["ply"]: r for r in rows}
    out: dict[int, MoveEval] = {}
    for ply in sorted(by_ply):
        row = by_ply[ply]
        if ply == 0:
            cp_before, mate_before = initial_cp, None
        else:
            prev = by_ply.get(ply - 1)
            if prev is None:
                continue
            cp_before, mate_before = prev["cp"], prev["mate"]
        if cp_before is None and mate_before is None:
            continue

        mover_is_white = (ply % 2 == 0)
        before = win_pct(cp=cp_before, mate=mate_before)
        after = win_pct(cp=row["cp"], mate=row["mate"])
        judgment = classify(win_pct_lost(before, after, mover_is_white))

        if cp_before is not None and row["cp"] is not None:
            signed_before = cp_before if mover_is_white else -cp_before
            signed_after = row["cp"] if mover_is_white else -row["cp"]
            cp_loss = max(0.0, float(signed_before - signed_after))
        else:
            cp_loss = None  # a mate score is involved; ACPL is meaningless there

        out[ply] = MoveEval(
            ply=ply,
            mover_is_white=mover_is_white,
            win_pct_before=before,
            win_pct_after=after,
            judgment=judgment,
            cp_loss=cp_loss,
            decided=(cp_before is not None and abs(cp_before) > DECIDED_CP)
                    or mate_before is not None,
        )
    return out


def judge_move(
    *,
    cp_before: int | None,
    mate_before: int | None,
    cp_after: int | None,
    mate_after: int | None,
    mover_is_white: bool,
) -> Judgment:
    """Classify one move from the evals bracketing it."""
    before = win_pct(cp=cp_before, mate=mate_before)
    after = win_pct(cp=cp_after, mate=mate_after)
    return classify(win_pct_lost(before, after, mover_is_white))
