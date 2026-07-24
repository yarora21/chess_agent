from __future__ import annotations

import pytest

from chess_agent import store


@pytest.fixture
def conn():
    connection = store.connect(":memory:")
    store.init_db(connection)
    yield connection
    connection.close()


def make_game(
    game_id: str = "abcd1234",
    *,
    created_at: int = 1_700_000_000_000,
    moves: str = "e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#",
    player_color: str = "white",
    **overrides,
) -> dict:
    """A Lichess export record shaped like the real NDJSON output (Scholar's mate)."""
    me = {"user": {"name": "MyName", "id": "myname"}, "rating": 1350, "ratingDiff": 8}
    them = {"user": {"name": "Rival", "id": "rival"}, "rating": 1372, "ratingDiff": -8}
    white, black = (me, them) if player_color == "white" else (them, me)
    record = {
        "id": game_id,
        "rated": True,
        "variant": "standard",
        "speed": "rapid",
        "perf": "rapid",
        "createdAt": created_at,
        "lastMoveAt": created_at + 500_000,
        "status": "mate",
        "players": {"white": white, "black": black},
        "winner": "white",
        "opening": {"eco": "C20", "name": "King's Pawn Game", "ply": 4},
        "moves": moves,
        "clocks": [60000, 60000, 59000, 58500, 57000, 57000, 55000],
        "clock": {"initial": 600, "increment": 0, "totalTime": 600},
        "division": {"middle": 4, "end": 6},
    }
    record.update(overrides)
    return record


@pytest.fixture
def game_record():
    return make_game()
