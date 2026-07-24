"""Phase 3: per-game metrics.

Dicts stand in for sqlite3.Row here -- both support row["col"], and the metrics
code only ever reads columns by name.
"""

from __future__ import annotations

from chess_agent import evals, metrics


def move(ply, *, phase="middlegame", is_player=1, book=0, clock=None):
    return {"ply": ply, "phase": phase, "is_player": is_player, "book": book,
            "clock_centis": clock, "san": "Nf3", "uci": "g1f3"}


def ev(ply, cp):
    return {"ply": ply, "cp": cp, "mate": None}


def compute(move_rows, eval_rows, initial_cp=20, result=None):
    return metrics.compute_game(move_rows, eval_rows, initial_cp, "g1", "e1", result)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_only_the_players_moves_are_counted():
    moves = [move(0, is_player=1), move(1, is_player=0), move(2, is_player=1)]
    # White hangs everything on ply 2; Black's ply 1 is irrelevant to the player.
    m = compute(moves, [ev(0, 20), ev(1, 20), ev(2, -600)])
    assert m.moves == 2
    assert m.blunders == 1


def test_opponent_blunders_are_not_credited_to_the_player():
    moves = [move(0, is_player=0), move(1, is_player=0)]
    m = compute(moves, [ev(0, 20), ev(1, 20)])
    assert m.moves == 0 and m.blunders == 0


def test_blunders_are_attributed_to_the_phase_they_happened_in():
    moves = [move(0, phase="opening"), move(2, phase="middlegame"),
             move(4, phase="endgame")]
    # Each blunder starts from a roughly level position. Starting from -600
    # instead would NOT register: from an already-lost position a further drop
    # costs little win probability, which is the whole point of thresholding on
    # win% rather than centipawns.
    m = compute(moves, [ev(0, 20), ev(1, 20), ev(2, -600), ev(3, 20), ev(4, -600)])
    assert (m.opening_moves, m.opening_blunders) == (1, 0)
    assert (m.middlegame_moves, m.middlegame_blunders) == (1, 1)
    assert (m.endgame_moves, m.endgame_blunders) == (1, 1)


def test_further_loss_from_an_already_lost_position_is_not_a_blunder():
    """-600 to -1400 is a huge centipawn drop but barely moves win probability."""
    m = compute([move(0)], [ev(0, -1400)] , initial_cp=-600)
    assert m.blunders == 0


def test_phase_move_counts_sum_to_total_moves():
    moves = [move(0, phase="opening"), move(2, phase="opening"),
             move(4, phase="middlegame"), move(6, phase="endgame")]
    m = compute(moves, [ev(p, 20) for p in range(8)])
    assert m.opening_moves + m.middlegame_moves + m.endgame_moves == m.moves


# ---------------------------------------------------------------------------
# Time trouble
# ---------------------------------------------------------------------------

def test_time_trouble_is_counted_below_the_threshold():
    below = metrics.TIME_TROUBLE_CENTIS - 1
    moves = [move(0, clock=below), move(2, clock=metrics.TIME_TROUBLE_CENTIS + 1)]
    m = compute(moves, [ev(0, 20), ev(1, 20), ev(2, -600)])
    assert m.time_trouble_moves == 1
    assert m.time_trouble_blunders == 0, "the blunder was played with time to spare"


def test_missing_clocks_never_count_as_time_trouble():
    m = compute([move(0, clock=None)], [ev(0, 20)])
    assert m.time_trouble_moves == 0


# ---------------------------------------------------------------------------
# ACPL
# ---------------------------------------------------------------------------

def test_acpl_excludes_already_decided_positions():
    # Starting from +1500 (decided), a drop to +1200 is engine noise, not skill.
    m = compute([move(0), move(2)], [ev(0, 1500), ev(1, 1500), ev(2, 1200)])
    assert m.acpl_moves == 1, "only the move from a non-decided position counts"


def test_acpl_is_capped():
    m = compute([move(0)], [ev(0, -5000)], initial_cp=20)
    assert m.acpl is not None and m.acpl <= metrics.ACPL_CAP


def test_acpl_is_none_when_nothing_was_measurable():
    m = compute([move(0, is_player=0)], [ev(0, 20)])
    assert m.acpl is None and m.acpl_moves == 0


def test_a_gain_is_not_a_loss():
    """Improving the position must not count as centipawn loss."""
    m = compute([move(0)], [ev(0, 300)], initial_cp=20)
    assert m.acpl == 0.0


# ---------------------------------------------------------------------------
# Book depth and conversion
# ---------------------------------------------------------------------------

def test_book_depth_is_the_first_non_book_ply():
    moves = [move(0, book=1), move(1, book=1, is_player=0), move(2, book=0)]
    m = compute(moves, [ev(p, 20) for p in range(3)])
    assert m.first_out_of_book_ply == 2


def test_book_depth_counts_opponent_plies_too():
    """Book depth describes the game, not the player -- both sides leave theory."""
    moves = [move(0, book=1), move(1, book=0, is_player=0), move(2, book=0)]
    m = compute(moves, [ev(p, 20) for p in range(3)])
    assert m.first_out_of_book_ply == 1


def test_conversion_requires_both_reaching_and_winning():
    winning = [move(0)], [ev(0, 900)]
    assert compute(*winning, result=1.0).converted == 1
    assert compute(*winning, result=0.0).converted == 0
    assert compute(*winning, result=0.0).reached_winning == 1


def test_never_winning_means_nothing_to_convert():
    m = compute([move(0)], [ev(0, 10)], result=1.0)
    assert m.reached_winning == 0 and m.converted == 0


def test_winning_as_black_is_detected():
    """A -900 eval is winning for Black; POV must not be hardcoded to White."""
    moves = [move(1, is_player=1)]  # ply 1 -> Black to have moved
    m = compute(moves, [ev(0, 20), ev(1, -900)], result=1.0)
    assert m.reached_winning == 1 and m.converted == 1


# ---------------------------------------------------------------------------
# Descriptive records
# ---------------------------------------------------------------------------

def test_descriptive_records_carry_their_sample_size():
    d = metrics.Descriptive("blunder rate", "endgame", 6.6, 5.7, 6.6, 700, 683, 5861, "%")
    record = d.as_record()
    assert record["first_n"] == 700 and record["total_n"] == 5861, (
        "sample size must travel with every number -- the MVP has no significance "
        "test, so the denominator is the only guard against over-reading"
    )


# ---------------------------------------------------------------------------
# Maia
# ---------------------------------------------------------------------------

def test_maia_match_is_scored_against_the_move_actually_played(conn):
    from chess_agent import maia, parse, store
    from conftest import make_game

    game, moves = parse.parse_game(make_game("mg1"), "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    played = [m for m in moves if m["is_player"]]

    rows = [("mg1", played[0]["ply"], 1200, played[0]["uci"], 1, "sha", "now"),
            ("mg1", played[1]["ply"], 1200, "a1a2", 0, "sha", "now")]
    conn.executemany(
        "INSERT INTO maia_moves (game_id, ply, rating, predicted_uci, matched, "
        "weights_sha, computed_at) VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()

    rates = maia.match_rates(conn, window=10)
    assert len(rates) == 1
    assert rates[0]["rating"] == 1200
    assert rates[0]["overall"] == 50.0 and rates[0]["total_n"] == 2


def test_discriminating_rates_ignore_positions_where_the_nets_agree(conn):
    """Agreed positions are obvious moves -- they carry no signal about level."""
    from chess_agent import maia, parse, store
    from conftest import make_game

    game, moves = parse.parse_game(make_game("mg2"), "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    p = [m for m in moves if m["is_player"]]

    rows = [
        # ply A: both nets predict the played move -> agreement, must be ignored
        ("mg2", p[0]["ply"], 1200, p[0]["uci"], 1, "s", "now"),
        ("mg2", p[0]["ply"], 1500, p[0]["uci"], 1, "s", "now"),
        # ply B: nets differ, player played the 1500 move
        ("mg2", p[1]["ply"], 1200, "a1a2", 0, "s", "now"),
        ("mg2", p[1]["ply"], 1500, p[1]["uci"], 1, "s", "now"),
    ]
    conn.executemany(
        "INSERT INTO maia_moves (game_id, ply, rating, predicted_uci, matched, "
        "weights_sha, computed_at) VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()

    d = maia.discriminating_rates(conn, 1200, 1500, window=10)
    assert d["overall"]["disagreements"] == 1, "the agreed position must be excluded"
    assert d["overall"]["played_high"] == 1 and d["overall"]["played_low"] == 0
    assert d["overall"]["high_share"] == 100.0
