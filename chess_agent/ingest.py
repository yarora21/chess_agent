"""Stream the player's full rapid history from Lichess into the store.

The whole history is ONE streamed request, not one request per game — a few
hundred games is seconds, and no rate limit is in play. The rate limit only
bites on per-game or per-position endpoints in a loop, which this project never
does (that is what local Stockfish is for).

berserk owns auth and the account lookup, but the export itself goes through a
raw streamed request: berserk 0.14's `export_by_player` does not expose the
`division` parameter, and division is exactly what gives us phase boundaries.
This is the documented "fall back to raw requests where berserk lags" path.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Callable, Iterator

import berserk
import requests

from . import config, openings, parse, store

EXPORT_URL = "https://lichess.org/api/games/user/{username}"

# Everything the pipeline needs, requested once.
EXPORT_PARAMS = {
    "rated": "true",
    "perfType": config.PERF_TYPE,  # rapid only — never aggregate across perf types
    "moves": "true",
    "pgnInJson": "true",
    "tags": "true",
    "clocks": "true",
    "evals": "true",       # free server evals where Lichess already analysed the game
    "opening": "true",
    "division": "true",    # phase boundaries
    "sort": "dateAsc",
}


def get_token() -> str:
    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        raise RuntimeError(
            "LICHESS_TOKEN is not set. Create a personal token at "
            "https://lichess.org/account/oauth/token and export it."
        )
    return token


def get_username(token: str) -> str:
    client = berserk.Client(session=berserk.TokenSession(token))
    return client.account.get()["username"]


def stream_games(
    username: str,
    token: str,
    since_ms: int | None = None,
    max_games: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield raw game records, oldest first, resuming across rate limits.

    On HTTP 429 we wait a full 60 s (limits are unpublished and dynamic) and
    resume the stream from the last game we actually received, so a retry never
    re-downloads the whole history.
    """
    params = dict(EXPORT_PARAMS)
    if max_games is not None:
        params["max"] = str(max_games)

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/x-ndjson"}
    cursor = since_ms
    yielded = 0
    attempts = 0

    while True:
        if cursor is not None:
            params["since"] = str(cursor + 1)
        if max_games is not None:
            remaining = max_games - yielded
            if remaining <= 0:
                return
            params["max"] = str(remaining)

        try:
            with requests.get(
                EXPORT_URL.format(username=username),
                params=params,
                headers=headers,
                stream=True,
                timeout=(10, 120),
            ) as response:
                if response.status_code == 429:
                    raise _RateLimited()
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    game = json.loads(line)
                    created = game.get("createdAt")
                    if isinstance(created, int):
                        cursor = created
                    yielded += 1
                    attempts = 0  # progress resets the retry budget
                    yield game
            return
        except (_RateLimited, requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            attempts += 1
            if attempts > config.MAX_RATE_LIMIT_RETRIES:
                raise RuntimeError(
                    f"giving up after {attempts} consecutive failures ({exc!r}); "
                    f"{yielded} games streamed so far"
                ) from exc
            time.sleep(config.RATE_LIMIT_BACKOFF_S)


class _RateLimited(Exception):
    pass


def ingest(
    conn: sqlite3.Connection,
    username: str | None = None,
    token: str | None = None,
    max_games: int | None = None,
    full: bool = False,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Pull rapid games into the store.

    Incremental by default: only games newer than the newest stored one are
    fetched. `full=True` re-streams everything (re-parsing in place; games are
    replaced, not duplicated).
    """
    token = token or get_token()
    username = username or get_username(token)

    store.init_db(conn)

    since_ms = None
    if not full:
        row = conn.execute("SELECT MAX(created_at) AS m FROM games").fetchone()
        since_ms = row["m"]

    known = store.existing_game_ids(conn)
    added = skipped = failed = 0
    failures: list[str] = []

    for game in stream_games(username, token, since_ms=since_ms, max_games=max_games):
        if game.get("perf") != config.PERF_TYPE:
            skipped += 1  # belt-and-braces: the API filter should already exclude these
            continue
        if game.get("variant") not in (None, "standard"):
            skipped += 1
            continue
        try:
            game_row, move_rows = parse.parse_game(game, username)
        except parse.ParseError as exc:
            failed += 1
            failures.append(str(exc))
            continue
        is_new = game_row["game_id"] not in known
        store.upsert_game(conn, game_row, move_rows)
        known.add(game_row["game_id"])
        added += is_new
        if on_progress:
            on_progress(added, skipped, failed)

    total = store.reindex_games(conn)
    store.set_meta(
        conn,
        username=username,
        openings_sha=openings.openings_sha(),
        last_ingest_at=store.utcnow(),
    )

    return {
        "username": username,
        "added": added,
        "skipped": skipped,
        "failed": failed,
        "failures": failures[:10],
        "total_games": total,
    }
