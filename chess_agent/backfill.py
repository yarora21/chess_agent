"""Parallel Stockfish backfill.

One single-threaded engine per physical core, one game per worker. Never one
engine with many threads: Lazy SMP scales node count near-linearly but
time-to-depth poorly, so N threads on one position is far slower than N
positions in parallel.

Workers compute and return; the parent process does all the writing. That keeps
SQLite to a single writer (no lock contention) and makes the run resumable --
games already evaluated under this engine config are skipped, so an interrupted
backfill picks up where it stopped.
"""

from __future__ import annotations

import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import chess
import chess.engine

from . import config, store

def _limit(settings: config.StockfishSettings) -> chess.engine.Limit:
    if settings.depth is not None and settings.movetime_ms is not None:
        raise ValueError("set exactly one of depth / movetime_ms")
    if settings.depth is not None:
        return chess.engine.Limit(depth=settings.depth)
    if settings.movetime_ms is not None:
        return chess.engine.Limit(time=settings.movetime_ms / 1000)
    raise ValueError("set one of depth / movetime_ms")


def _open_engine(settings: config.StockfishSettings) -> chess.engine.SimpleEngine:
    """Start one engine. Callers MUST quit it -- see the warning below.

    Do not be tempted to cache this in a module-level global to save the ~50ms
    startup. python-chess runs the engine on a non-daemon thread, so a worker
    process holding a live engine cannot exit, and ProcessPoolExecutor deadlocks
    on shutdown with the engines idling at 0% CPU. That bug cost a 20-minute
    hang here. 50ms per game against 724 games is ~5s of wall clock across the
    pool; correctness is worth vastly more than that.
    """
    engine = chess.engine.SimpleEngine.popen_uci(settings.binary)
    engine.configure({"Threads": settings.threads, "Hash": settings.hash_mb})
    return engine


def engine_version(binary: str) -> str:
    engine = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        return engine.id.get("name", "unknown")
    finally:
        engine.quit()


def _score_to_cols(score: chess.engine.PovScore) -> tuple[int | None, int | None]:
    """White-POV (cp, mate); exactly one is non-None, matching the schema CHECK."""
    white = score.white()
    if white.is_mate():
        return None, white.mate()
    return white.score(), None


@dataclass
class GameEvals:
    game_id: str
    rows: list[dict[str, Any]]
    elapsed_s: float
    positions: int


def _analyse_game(args: tuple[str, list[str], str, dict]) -> GameEvals:
    """Worker entry point. Replays the game and evaluates the position after each ply.

    Evaluating the position *after* each ply matches how Lichess aligns its own
    analysis array, which is what makes the phase-2 validation a like-for-like
    comparison.
    """
    game_id, ucis, binary, settings_dict = args
    settings = config.StockfishSettings(**settings_dict)
    limit = _limit(settings)

    board = chess.Board()
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    engine = _open_engine(settings)
    try:
        for ply, uci in enumerate(ucis):
            board.push(chess.Move.from_uci(uci))
            if board.is_game_over(claim_draw=False):
                # Terminal position: the result is known, an eval would be noise.
                # Lichess omits these too (its analysis array is one short on mates).
                break
            info = engine.analyse(board, limit)
            cp, mate = _score_to_cols(info["score"])
            pv = info.get("pv") or []
            rows.append(
                {
                    "game_id": game_id,
                    "ply": ply,
                    "cp": cp,
                    "mate": mate,
                    "best_move": pv[0].uci() if pv else None,
                    "pv": " ".join(m.uci() for m in pv[:8]) or None,
                    "depth_reached": info.get("depth"),
                    "nodes": info.get("nodes"),
                }
            )
    finally:
        engine.quit()
    return GameEvals(game_id, rows, time.perf_counter() - started, len(rows))


def analyse_initial_position(binary: str, settings: config.StockfishSettings) -> int | None:
    """Eval of the starting position: the 'before' side of every game's first move."""
    engine = chess.engine.SimpleEngine.popen_uci(binary)
    try:
        engine.configure({"Threads": settings.threads, "Hash": settings.hash_mb})
        info = engine.analyse(chess.Board(), _limit(settings))
        cp, _ = _score_to_cols(info["score"])
        return cp
    finally:
        engine.quit()


def register(conn: sqlite3.Connection, settings: config.StockfishSettings) -> str:
    """Register this engine identity and record its starting-position eval."""
    version = engine_version(settings.binary)
    engine_id = store.register_engine_config(
        conn,
        provenance="local",
        engine_name="stockfish",
        engine_version=version,
        depth=settings.depth,
        movetime_ms=settings.movetime_ms,
        threads=settings.threads,
        extra={"hash_mb": settings.hash_mb},
    )
    row = conn.execute(
        "SELECT initial_cp FROM engine_configs WHERE engine_id = ?", (engine_id,)
    ).fetchone()
    if row["initial_cp"] is None:
        with conn:
            conn.execute(
                "UPDATE engine_configs SET initial_cp = ? WHERE engine_id = ?",
                (analyse_initial_position(settings.binary, settings), engine_id),
            )
    return engine_id


def pending_games(conn: sqlite3.Connection, engine_id: str) -> list[tuple[str, list[str]]]:
    """Cohort games with no evals yet under this engine config."""
    lo, hi = store.cohort_bounds()
    rows = conn.execute(
        f"SELECT g.game_id FROM games g WHERE {store.COHORT_WHERE} "
        "AND NOT EXISTS (SELECT 1 FROM evals e WHERE e.game_id = g.game_id "
        "                AND e.engine_id = ?) ORDER BY g.game_index",
        (lo, hi, engine_id),
    ).fetchall()
    out = []
    for r in rows:
        ucis = [m["uci"] for m in store.get_moves(conn, r["game_id"])]
        out.append((r["game_id"], ucis))
    return out


def write_evals(conn: sqlite3.Connection, engine_id: str, result: GameEvals,
                table: str = "evals", **extra: Any) -> None:
    cols = ["game_id", "ply", "engine_id", "cp", "mate", "best_move", "pv",
            "depth_reached", "nodes", "computed_at"] + list(extra)
    now = store.utcnow()
    with conn:
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
            f"VALUES ({', '.join('?' for _ in cols)})",
            [
                tuple([r["game_id"], r["ply"], engine_id, r["cp"], r["mate"],
                       r["best_move"], r["pv"], r["depth_reached"], r["nodes"], now]
                      + list(extra.values()))
                for r in result.rows
            ],
        )


def run(
    conn: sqlite3.Connection,
    settings: config.StockfishSettings | None = None,
    games: Sequence[tuple[str, list[str]]] | None = None,
    table: str = "evals",
    engine_id: str | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    settings = settings or config.STOCKFISH
    engine_id = engine_id or register(conn, settings)
    todo = list(games) if games is not None else pending_games(conn, engine_id)
    workers = settings.workers or os.cpu_count() or 4

    started = time.perf_counter()
    done = positions = 0
    settings_dict = settings.__dict__.copy()

    if not todo:
        return {"engine_id": engine_id, "games": 0, "positions": 0, "elapsed_s": 0.0}

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_analyse_game, (gid, ucis, settings.binary, settings_dict)): gid
            for gid, ucis in todo
        }
        for future in as_completed(futures):
            result = future.result()
            write_evals(conn, engine_id, result, table=table, **extra)
            done += 1
            positions += result.positions
            if on_progress:
                on_progress(done, len(todo), time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    return {
        "engine_id": engine_id,
        "games": done,
        "positions": positions,
        "elapsed_s": elapsed,
        "positions_per_s": positions / elapsed if elapsed else 0,
        "workers": workers,
    }
