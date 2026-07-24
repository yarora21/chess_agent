"""Turn a raw Lichess export record into `games` and `moves` rows.

This is the only place that knows the shape of the Lichess JSON. Everything
downstream reads the store.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import chess

from . import openings


class ParseError(ValueError):
    """A game record could not be turned into rows (bad SAN, missing moves, ...)."""


def _to_ms(value: Any) -> int | None:
    """Lichess sends epoch-ms ints; berserk converts some of them to datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return int(value.timestamp() * 1000)
    return int(value)


def _json_default(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def _player_side(game: dict[str, Any], username: str) -> str:
    lowered = username.lower()
    for color in ("white", "black"):
        user = (game.get("players", {}).get(color, {}) or {}).get("user") or {}
        name = user.get("id") or user.get("name") or ""
        if name.lower() == lowered:
            return color
    raise ParseError(f"{username} is not a player in game {game.get('id')}")


def _phase_for_ply(ply: int, middle: int | None, end: int | None) -> str:
    """Phase of a 0-based ply from the export's `division` field.

    Convention: `division.middle` / `division.end` are ply counts, so move index
    i is middlegame once i >= middle and endgame once i >= end. A missing field
    means the game never reached that phase.
    """
    if end is not None and ply >= end:
        return "endgame"
    if middle is not None and ply >= middle:
        return "middlegame"
    return "opening"


def parse_game(game: dict[str, Any], username: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (game_row, move_rows). Raises ParseError on anything unreplayable."""
    game_id = game.get("id")
    if not game_id:
        raise ParseError("record has no game id")

    moves_san = (game.get("moves") or "").strip()
    if not moves_san:
        raise ParseError(f"game {game_id} has no moves")

    player_color = _player_side(game, username)
    opponent_color = "black" if player_color == "white" else "white"
    players = game.get("players", {}) or {}

    def side_info(color: str) -> tuple[str | None, int | None, int | None]:
        p = players.get(color, {}) or {}
        user = p.get("user") or {}
        name = user.get("name") or user.get("id")
        if name is None and p.get("aiLevel"):
            name = f"stockfish-level-{p['aiLevel']}"
        return name, p.get("rating"), p.get("ratingDiff")

    white_name, white_rating, _ = side_info("white")
    black_name, black_rating, _ = side_info("black")
    _, player_rating, player_rating_diff = side_info(player_color)
    opponent_name, opponent_rating, opponent_rating_diff = side_info(opponent_color)

    winner = game.get("winner")
    if winner is None:
        player_result = 0.5
    else:
        player_result = 1.0 if winner == player_color else 0.0

    clock = game.get("clock") or {}
    division = game.get("division") or {}
    middle, end = division.get("middle"), division.get("end")
    opening = game.get("opening") or {}
    clocks = game.get("clocks") or []
    increment_centis = (clock.get("increment") or 0) * 100
    initial_centis = (clock.get("initial") or 0) * 100

    board = chess.Board()
    move_rows: list[dict[str, Any]] = []

    for ply, san in enumerate(moves_san.split()):
        fen_before = board.fen()
        try:
            move = board.parse_san(san)
        except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError) as exc:
            raise ParseError(f"game {game_id} ply {ply}: cannot replay SAN {san!r}: {exc}") from exc
        uci = move.uci()
        board.push(move)
        epd_after = board.epd()

        side = "white" if ply % 2 == 0 else "black"
        book_hit = openings.lookup(epd_after)

        clock_centis = clocks[ply] if ply < len(clocks) else None
        time_spent = None
        if clock_centis is not None:
            if ply < 2:
                # First move of each side: the clock starts at the initial time and
                # Lichess does not credit an increment before the first move.
                time_spent = initial_centis - clock_centis
            else:
                time_spent = clocks[ply - 2] - clock_centis + increment_centis

        move_rows.append(
            {
                "game_id": game_id,
                "ply": ply,
                "move_number": ply // 2 + 1,
                "side": side,
                "is_player": int(side == player_color),
                "san": san,
                "uci": uci,
                "fen_before": fen_before,
                "epd_after": epd_after,
                "clock_centis": clock_centis,
                "time_spent_centis": time_spent,
                "phase": _phase_for_ply(ply, middle, end),
                "book": int(book_hit is not None),
                "book_eco": book_hit[0] if book_hit else None,
                "book_name": book_hit[1] if book_hit else None,
            }
        )

    game_row = {
        "game_id": game_id,
        "rated": int(bool(game.get("rated"))),
        "perf": game.get("perf"),
        "speed": game.get("speed"),
        "variant": game.get("variant"),
        "status": game.get("status"),
        "winner": winner,
        "created_at": _to_ms(game.get("createdAt")),
        "last_move_at": _to_ms(game.get("lastMoveAt")),
        "player_color": player_color,
        "player_rating": player_rating,
        "player_rating_diff": player_rating_diff,
        "player_result": player_result,
        "opponent_name": opponent_name,
        "opponent_rating": opponent_rating,
        "opponent_rating_diff": opponent_rating_diff,
        "white_name": white_name,
        "white_rating": white_rating,
        "black_name": black_name,
        "black_rating": black_rating,
        "eco": opening.get("eco"),
        "opening_name": opening.get("name"),
        "opening_ply": opening.get("ply"),
        "clock_initial": clock.get("initial"),
        "clock_increment": clock.get("increment"),
        "clock_total_moves": clock.get("totalTime"),
        "division_middlegame": middle,
        "division_endgame": end,
        "n_plies": len(move_rows),
        "has_clocks": int(bool(clocks)),
        "has_lichess_analysis": int(bool(game.get("analysis"))),
        "moves_san": moves_san,
        "pgn": game.get("pgn"),
        "raw_json": json.dumps(game, default=_json_default, sort_keys=True),
    }
    return game_row, move_rows
