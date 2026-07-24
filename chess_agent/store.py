"""SQLite store. Every pipeline stage reads and writes through this module.

Crash-safety matters here: the phase-2 backfill writes from many worker
processes, so the connection is opened in WAL mode with a busy timeout.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import config


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path is not None else config.DB_PATH
    if path != Path(":memory:"):
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema if absent. Idempotent."""
    conn.executescript(config.SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT OR IGNORE INTO meta (id, schema_version, perf_type) VALUES (1, ?, ?)",
        (config.SCHEMA_VERSION, config.PERF_TYPE),
    )
    conn.commit()
    _migrate(conn)
    _check_schema_version(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring an older store up to the current schema. Each step is idempotent."""
    row = conn.execute("SELECT schema_version FROM meta WHERE id = 1").fetchone()
    version = row["schema_version"] if row else config.SCHEMA_VERSION

    if version < 2:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(engine_configs)")}
        if "initial_cp" not in cols:
            conn.execute("ALTER TABLE engine_configs ADD COLUMN initial_cp INTEGER")
        version = 2

    if version != (row["schema_version"] if row else version):
        conn.execute("UPDATE meta SET schema_version = ? WHERE id = 1", (version,))
    conn.commit()


def _check_schema_version(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT schema_version FROM meta WHERE id = 1").fetchone()
    if row and row["schema_version"] != config.SCHEMA_VERSION:
        raise RuntimeError(
            f"Store was built with schema v{row['schema_version']}, code expects "
            f"v{config.SCHEMA_VERSION}. See the migration notes in schema.sql."
        )


def set_meta(conn: sqlite3.Connection, **fields: Any) -> None:
    if not fields:
        return
    assignments = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE meta SET {assignments} WHERE id = 1", tuple(fields.values()))
    conn.commit()


def get_meta(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute("SELECT * FROM meta WHERE id = 1").fetchone()
    return dict(row) if row else {}


def existing_game_ids(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT game_id FROM games")}


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

_GAME_COLUMNS = (
    "game_id", "rated", "perf", "speed", "variant", "status", "winner",
    "created_at", "last_move_at", "player_color", "player_rating",
    "player_rating_diff", "player_result", "opponent_name", "opponent_rating",
    "opponent_rating_diff", "white_name", "white_rating", "black_name",
    "black_rating", "eco", "opening_name", "opening_ply", "clock_initial",
    "clock_increment", "clock_total_moves", "division_middlegame",
    "division_endgame", "n_plies", "has_clocks", "has_lichess_analysis",
    "moves_san", "pgn", "raw_json", "ingested_at",
)

_MOVE_COLUMNS = (
    "game_id", "ply", "move_number", "side", "is_player", "san", "uci",
    "fen_before", "epd_after", "clock_centis", "time_spent_centis", "phase",
    "book", "book_eco", "book_name",
)


def upsert_game(conn: sqlite3.Connection, game: dict[str, Any], moves: Sequence[dict[str, Any]]) -> None:
    """Write one game and all its moves in a single transaction.

    Re-ingesting a game replaces its moves wholesale, so a re-parse after a
    parser fix can never leave a half-old, half-new move list behind.
    """
    game = {**game, "ingested_at": utcnow()}
    placeholders = ", ".join("?" for _ in _GAME_COLUMNS)
    with conn:  # transaction
        conn.execute(
            f"INSERT OR REPLACE INTO games ({', '.join(_GAME_COLUMNS)}) VALUES ({placeholders})",
            tuple(game.get(c) for c in _GAME_COLUMNS),
        )
        conn.execute("DELETE FROM moves WHERE game_id = ?", (game["game_id"],))
        conn.executemany(
            f"INSERT INTO moves ({', '.join(_MOVE_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _MOVE_COLUMNS)})",
            [tuple(m.get(c) for c in _MOVE_COLUMNS) for m in moves],
        )


def reindex_games(conn: sqlite3.Connection) -> int:
    """Assign game_index: 0-based chronological position, ordered by created_at.

    This is the project's trend regressor, so it is recomputed over the whole
    table after every ingest -- a backfill of older games must renumber
    everything, not append. Ties on created_at break by game_id for determinism.
    """
    rows = conn.execute(
        "SELECT game_id FROM games ORDER BY created_at ASC, game_id ASC"
    ).fetchall()
    with conn:
        # Two passes: clear first, since game_index carries a UNIQUE constraint
        # and a partial renumbering would collide mid-update.
        conn.execute("UPDATE games SET game_index = NULL")
        conn.executemany(
            "UPDATE games SET game_index = ? WHERE game_id = ?",
            [(i, r["game_id"]) for i, r in enumerate(rows)],
        )
    return len(rows)


def register_engine_config(
    conn: sqlite3.Connection,
    *,
    provenance: str,
    engine_name: str,
    engine_version: str | None = None,
    depth: int | None = None,
    movetime_ms: int | None = None,
    threads: int | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Register an engine identity and return its engine_id.

    The id is a hash of the identity fields, so the same settings always resolve
    to the same id and different settings can never silently share a cache.
    """
    extra_json = json.dumps(extra or {}, sort_keys=True, separators=(",", ":"))
    identity = json.dumps(
        {
            "provenance": provenance,
            "engine_name": engine_name,
            "engine_version": engine_version,
            "depth": depth,
            "movetime_ms": movetime_ms,
            "threads": threads,
            "extra": extra_json,
        },
        sort_keys=True,
    )
    engine_id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO engine_configs (engine_id, provenance, engine_name, "
            "engine_version, depth, movetime_ms, threads, extra_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (engine_id, provenance, engine_name, engine_version, depth,
             movetime_ms, threads, extra_json, utcnow()),
        )
    return engine_id


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def iter_games(conn: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    yield from conn.execute("SELECT * FROM games ORDER BY game_index")


# The cohort filter. Metrics, claims and narration must go through these --
# reading `games` directly would silently pull in the stale and provisional
# games that config.ANALYSIS_* exists to exclude.
COHORT_WHERE = "game_index BETWEEN ? AND ?"


def cohort_bounds() -> tuple[int, int]:
    return config.ANALYSIS_START_INDEX, config.ANALYSIS_END_INDEX


def analysis_games(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT * FROM games WHERE {COHORT_WHERE} ORDER BY game_index", cohort_bounds()
    ).fetchall()


def analysis_moves(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        f"SELECT m.* FROM moves m JOIN games g USING (game_id) WHERE {COHORT_WHERE} "
        "ORDER BY g.game_index, m.ply",
        cohort_bounds(),
    ).fetchall()


def cohort_summary(conn: sqlite3.Connection) -> dict[str, Any]:
    lo, hi = cohort_bounds()
    row = conn.execute(
        f"SELECT COUNT(*) n, MIN(player_rating) lo_r, MAX(player_rating) hi_r, "
        f"SUM(n_plies) plies, SUM(has_lichess_analysis) analysed "
        f"FROM games WHERE {COHORT_WHERE}",
        (lo, hi),
    ).fetchone()
    excluded = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] - row["n"]
    return {
        "cohort_range": f"{lo}-{hi}",
        "cohort_games": row["n"],
        "cohort_moves": row["plies"] or 0,
        "excluded_games": excluded,
        "rating_min": row["lo_r"],
        "rating_max": row["hi_r"],
        "with_lichess_analysis": row["analysed"] or 0,
    }


def get_moves(conn: sqlite3.Connection, game_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM moves WHERE game_id = ? ORDER BY ply", (game_id,)
    ).fetchall()


def summary(conn: sqlite3.Connection) -> dict[str, Any]:
    g = conn.execute(
        "SELECT COUNT(*) n, MIN(created_at) first, MAX(created_at) last, "
        "SUM(has_clocks) with_clocks, SUM(has_lichess_analysis) analysed FROM games"
    ).fetchone()
    m = conn.execute("SELECT COUNT(*) n FROM moves").fetchone()
    ratings = conn.execute(
        "SELECT player_rating FROM games WHERE player_rating IS NOT NULL ORDER BY game_index"
    ).fetchall()
    return {
        "games": g["n"],
        "moves": m["n"],
        "first_game_at": g["first"],
        "last_game_at": g["last"],
        "games_with_clocks": g["with_clocks"] or 0,
        "games_with_lichess_analysis": g["analysed"] or 0,
        "first_rating": ratings[0][0] if ratings else None,
        "last_rating": ratings[-1][0] if ratings else None,
    }
