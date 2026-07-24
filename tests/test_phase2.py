"""Phase 2: win% conversion and move classification.

The point-of-view sign gets the most attention here. It is the one bug in this
layer that would produce a fully self-consistent, completely backwards report.
"""

from __future__ import annotations

import pytest

from chess_agent import evals


# ---------------------------------------------------------------------------
# Win% conversion
# ---------------------------------------------------------------------------

def test_equal_position_is_fifty_percent():
    assert evals.win_pct(cp=0) == pytest.approx(50.0)


def test_win_pct_is_monotonic_in_white_favour():
    values = [evals.win_pct(cp=cp) for cp in (-800, -300, -100, 0, 100, 300, 800)]
    assert values == sorted(values)
    assert 0 < values[0] < 10 and 90 < values[-1] < 100


def test_win_pct_is_symmetric_about_zero():
    for cp in (50, 200, 600):
        assert evals.win_pct(cp=cp) + evals.win_pct(cp=-cp) == pytest.approx(100.0)


def test_mate_saturates():
    assert evals.win_pct(mate=1) == 100.0
    assert evals.win_pct(mate=5) == 100.0
    assert evals.win_pct(mate=-1) == 0.0


def test_win_pct_requires_a_score():
    with pytest.raises(ValueError):
        evals.win_pct()


# ---------------------------------------------------------------------------
# Point of view -- the sign bug this project cannot afford
# ---------------------------------------------------------------------------

def test_white_losing_ground_is_a_loss_for_white():
    # +300 -> -300 with White to have moved: White threw it away.
    lost = evals.win_pct_lost(evals.win_pct(cp=300), evals.win_pct(cp=-300),
                              mover_is_white=True)
    assert lost > 0


def test_the_same_swing_is_a_gain_for_black():
    # The identical eval change, but Black made the move: Black gained.
    lost = evals.win_pct_lost(evals.win_pct(cp=300), evals.win_pct(cp=-300),
                              mover_is_white=False)
    assert lost < 0


def test_black_blundering_is_detected_as_black_losing_ground():
    # -300 (Black winning) -> +300, played by Black: a Black blunder.
    judgment = evals.judge_move(cp_before=-300, mate_before=None,
                                cp_after=300, mate_after=None, mover_is_white=False)
    assert judgment.is_blunder


def test_a_good_move_is_never_a_blunder_for_either_colour():
    for white in (True, False):
        before, after = (100, 120) if white else (-100, -120)
        j = evals.judge_move(cp_before=before, mate_before=None,
                             cp_after=after, mate_after=None, mover_is_white=white)
        assert j.name == "ok"


def test_walking_into_mate_is_a_blunder():
    j = evals.judge_move(cp_before=50, mate_before=None,
                         cp_after=None, mate_after=-2, mover_is_white=True)
    assert j.is_blunder and j.win_pct_lost > 40


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

def test_classification_bands_are_ordered():
    assert (evals.INACCURACY_WIN_PCT < evals.MISTAKE_WIN_PCT < evals.BLUNDER_WIN_PCT)


@pytest.mark.parametrize("lost,expected", [
    (0.0, "ok"), (4.9, "ok"),
    (5.0, "inaccuracy"), (9.9, "inaccuracy"),
    (10.0, "mistake"), (14.9, "mistake"),
    (15.0, "blunder"), (80.0, "blunder"),
])
def test_thresholds_are_inclusive_lower_bounds(lost, expected):
    assert evals.classify(lost).name == expected


def test_a_gain_is_not_a_blunder():
    assert evals.classify(-20.0).name == "ok"


def test_blunder_threshold_matches_the_value_fitted_to_lichess():
    """Derived from 21,157 judged plies in this store; do not drift without refitting."""
    assert evals.BLUNDER_WIN_PCT == 15.0


# ---------------------------------------------------------------------------
# Already-decided positions
# ---------------------------------------------------------------------------

def test_swings_in_decided_positions_are_still_large_but_flagged_by_cap():
    # +900 -> +700 is a big centipawn drop but barely moves win probability.
    j = evals.judge_move(cp_before=900, mate_before=None,
                         cp_after=700, mate_after=None, mover_is_white=True)
    assert j.name == "ok", "win% thresholds must not punish noise in won positions"
    assert evals.DECIDED_CP == 800
