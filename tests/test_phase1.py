"""Phase 1 gate: ingest, parsing, and storage produce a store that reconstructs."""

from __future__ import annotations

import chess
import pytest

from chess_agent import openings, parse, store, verify
from conftest import make_game


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_parse_produces_one_row_per_ply(game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    assert game["n_plies"] == 7 == len(moves)
    assert [m["ply"] for m in moves] == list(range(7))
    assert moves[0]["fen_before"] == chess.STARTING_FEN
    assert moves[0]["san"] == "e4" and moves[0]["uci"] == "e2e4"


def test_every_stored_position_is_the_one_actually_reached(game_record):
    _, moves = parse.parse_game(game_record, "MyName")
    board = chess.Board()
    for m in moves:
        assert board.fen() == m["fen_before"]
        board.push(chess.Move.from_uci(m["uci"]))
        assert board.epd() == m["epd_after"]
    assert board.is_checkmate()


def test_player_identity_follows_color(game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    assert game["player_color"] == "white"
    assert game["player_rating"] == 1350 and game["opponent_rating"] == 1372
    assert game["player_result"] == 1.0
    assert all(m["is_player"] == (m["side"] == "white") for m in moves)


def test_player_identity_when_playing_black():
    record = make_game(player_color="black")
    game, moves = parse.parse_game(record, "MyName")
    assert game["player_color"] == "black"
    assert game["player_rating"] == 1350
    assert game["player_result"] == 0.0  # white delivered mate
    assert all(m["is_player"] == (m["side"] == "black") for m in moves)


def test_username_matching_is_case_insensitive(game_record):
    game, _ = parse.parse_game(game_record, "myNAME")
    assert game["player_color"] == "white"


def test_unknown_player_is_a_parse_error(game_record):
    with pytest.raises(parse.ParseError):
        parse.parse_game(game_record, "SomeoneElse")


def test_unreplayable_san_is_a_parse_error():
    with pytest.raises(parse.ParseError):
        parse.parse_game(make_game(moves="e4 e5 Qxf7#"), "MyName")


# ---------------------------------------------------------------------------
# Phase boundaries
# ---------------------------------------------------------------------------

def test_phase_split_follows_division(game_record):
    _, moves = parse.parse_game(game_record, "MyName")  # middle=4, end=6
    assert [m["phase"] for m in moves] == [
        "opening", "opening", "opening", "opening",
        "middlegame", "middlegame",
        "endgame",
    ]


def test_missing_division_means_the_game_never_left_the_opening():
    record = make_game()
    del record["division"]
    _, moves = parse.parse_game(record, "MyName")
    assert {m["phase"] for m in moves} == {"opening"}


def test_endgame_never_reached():
    record = make_game(division={"middle": 4})
    _, moves = parse.parse_game(record, "MyName")
    assert [m["phase"] for m in moves].count("endgame") == 0
    assert [m["phase"] for m in moves][-1] == "middlegame"


# ---------------------------------------------------------------------------
# Clocks
# ---------------------------------------------------------------------------

def test_time_spent_uses_the_same_side_previous_clock(game_record):
    _, moves = parse.parse_game(game_record, "MyName")
    # clocks: [60000, 60000, 59000, 58500, 57000, 57000, 55000], initial 600s = 60000cs
    assert moves[0]["time_spent_centis"] == 0        # 60000 initial -> 60000
    assert moves[2]["time_spent_centis"] == 1000     # white: 60000 -> 59000
    assert moves[3]["time_spent_centis"] == 1500     # black: 60000 -> 58500
    assert moves[4]["time_spent_centis"] == 2000     # white: 59000 -> 57000


def test_increment_is_credited_after_the_first_move():
    record = make_game(clock={"initial": 600, "increment": 5, "totalTime": 600})
    _, moves = parse.parse_game(record, "MyName")
    assert moves[2]["time_spent_centis"] == 1000 + 500


def test_absent_clocks_are_null_not_zero():
    record = make_game()
    del record["clocks"]
    game, moves = parse.parse_game(record, "MyName")
    assert game["has_clocks"] == 0
    assert all(m["clock_centis"] is None and m["time_spent_centis"] is None for m in moves)


# ---------------------------------------------------------------------------
# Book tagging
# ---------------------------------------------------------------------------

def test_opening_book_loads():
    positions = openings.book_positions()
    assert len(positions) > 5_000
    board = chess.Board()
    board.push_san("e4")
    assert openings.lookup(board.epd()) is not None


def test_book_moves_are_tagged(game_record):
    _, moves = parse.parse_game(game_record, "MyName")
    assert moves[0]["book"] == 1, "1. e4 must be in the book"
    assert moves[0]["book_eco"] and moves[0]["book_name"]


def test_transposing_back_into_book_is_tagged_not_rejected():
    """Book is a per-position property, not a prefix -- real games transpose.

    1.d4 d5 2.c4 Nf6 3.g3 dxc4 4.Bg2 e6 leaves the named lines and re-enters
    the Catalan. Observed in real data (game tR3sRemz); an earlier version of
    the gate wrongly failed it.
    """
    record = make_game(moves="d4 d5 c4 Nf6 g3 dxc4 Bg2 e6 Nf3 Qd5", status="resign",
                       division={"middle": 8}, clocks=[])
    _, moves = parse.parse_game(record, "MyName")
    flags = [m["book"] for m in moves]
    assert flags[:4] == [1, 1, 1, 1]
    assert flags[4:7] == [0, 0, 0], "leaves known theory"
    assert flags[7] == 1, "transposes back into the Catalan"
    assert moves[7]["book_name"] and "Catalan" in moves[7]["book_name"]


def test_leaving_the_book_is_detectable(game_record):
    _, moves = parse.parse_game(game_record, "MyName")
    depth = sum(m["book"] for m in moves)
    assert 0 < depth < len(moves), "Scholar's mate leaves known theory before it ends"
    assert moves[depth]["book"] == 0 and moves[depth]["book_name"] is None


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_roundtrip_through_sqlite(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    stored = store.get_moves(conn, game["game_id"])
    assert len(stored) == len(moves)
    assert [r["san"] for r in stored] == [m["san"] for m in moves]
    assert stored[0]["fen_before"] == chess.STARTING_FEN


def test_reingesting_replaces_moves_rather_than_duplicating(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    store.upsert_game(conn, game, moves)
    assert len(store.get_moves(conn, game["game_id"])) == len(moves)
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1


def test_game_index_is_chronological_and_dense(conn):
    times = [3_000, 1_000, 2_000]
    for i, t in enumerate(times):
        game, moves = parse.parse_game(make_game(f"game{i}", created_at=t), "MyName")
        store.upsert_game(conn, game, moves)
    assert store.reindex_games(conn) == 3
    rows = conn.execute("SELECT game_id, game_index FROM games ORDER BY game_index").fetchall()
    assert [r["game_id"] for r in rows] == ["game1", "game2", "game0"]
    assert [r["game_index"] for r in rows] == [0, 1, 2]


def test_reindex_renumbers_when_older_games_arrive_later(conn):
    """A backfill of older games must renumber, not append — game_index is the regressor."""
    first, moves = parse.parse_game(make_game("newer", created_at=5_000), "MyName")
    store.upsert_game(conn, first, moves)
    store.reindex_games(conn)
    older, moves = parse.parse_game(make_game("older", created_at=1_000), "MyName")
    store.upsert_game(conn, older, moves)
    store.reindex_games(conn)
    rows = dict(conn.execute("SELECT game_id, game_index FROM games").fetchall())
    assert rows == {"older": 0, "newer": 1}


def test_engine_config_id_is_stable_and_settings_sensitive(conn):
    a = store.register_engine_config(conn, provenance="local", engine_name="stockfish",
                                     engine_version="17.1", movetime_ms=100, threads=1)
    same = store.register_engine_config(conn, provenance="local", engine_name="stockfish",
                                        engine_version="17.1", movetime_ms=100, threads=1)
    deeper = store.register_engine_config(conn, provenance="local", engine_name="stockfish",
                                          engine_version="17.1", movetime_ms=300, threads=1)
    assert a == same, "same settings must reuse the same eval cache"
    assert a != deeper, "different settings must never share an eval cache"
    assert conn.execute("SELECT COUNT(*) FROM engine_configs").fetchone()[0] == 2


def test_evals_reject_rows_with_both_cp_and_mate(conn, game_record):
    import sqlite3

    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    engine_id = store.register_engine_config(conn, provenance="local", engine_name="stockfish")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO evals (game_id, ply, engine_id, cp, mate, computed_at) "
            "VALUES (?, 0, ?, 35, 3, 'now')",
            (game["game_id"], engine_id),
        )


def test_deleting_a_game_cascades_to_moves(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    conn.execute("DELETE FROM games WHERE game_id = ?", (game["game_id"],))
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------

def test_verify_passes_on_a_clean_store(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    report = verify.verify_store(conn)
    assert report.ok, report.render()
    assert report.games_checked == 1 and report.games_ok == 1


def test_verify_catches_a_corrupted_position(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    conn.execute("UPDATE moves SET fen_before = ? WHERE game_id = ? AND ply = 3",
                 (chess.STARTING_FEN, game["game_id"]))
    conn.commit()
    report = verify.verify_store(conn)
    assert not report.ok
    assert "fen_before does not match replay" in report.render()


def test_verify_catches_a_wrong_move(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    conn.execute("UPDATE moves SET uci = 'a1a8' WHERE game_id = ? AND ply = 2",
                 (game["game_id"],))
    conn.commit()
    report = verify.verify_store(conn)
    assert not report.ok and "illegal" in report.render()


def test_verify_catches_phase_going_backwards(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    conn.execute("UPDATE moves SET phase = 'opening' WHERE game_id = ? AND ply = 6",
                 (game["game_id"],))
    conn.commit()
    report = verify.verify_store(conn)
    assert not report.ok and "phase goes backwards" in report.render()


def test_verify_catches_a_result_that_contradicts_the_board(conn, game_record):
    game, moves = parse.parse_game(game_record, "MyName")
    game["winner"] = "black"  # white delivered mate
    store.upsert_game(conn, game, moves)
    store.reindex_games(conn)
    report = verify.verify_store(conn)
    assert not report.ok and "is the side mated" in report.render()


def test_verify_catches_a_missing_reindex(conn):
    for i, t in enumerate([1_000, 2_000]):
        game, moves = parse.parse_game(make_game(f"g{i}", created_at=t), "MyName")
        store.upsert_game(conn, game, moves)
    # deliberately skip reindex_games
    report = verify.verify_store(conn)
    assert not report.ok
