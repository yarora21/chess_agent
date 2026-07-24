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
