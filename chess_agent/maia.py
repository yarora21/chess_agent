"""Maia move-matching: would a human at rating X have played what you played?

This is the opponent-independent signal in the project. Raw blunder rates rise
as opponents get stronger even when the player is improving, so they can only
ever be supporting evidence. "Do I now play the moves a 1500 plays" does not
depend on who was sitting across the board.

Queried at nodes=1: the policy head only, no search. Maia is being asked what a
human would play, not what is best -- running search would destroy the thing
that makes it useful.

Known approximation, on the record: Maia-1 was trained mostly on blitz and we
analyse rapid. Accepted at this rating band.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence

import chess
import chess.engine

from . import config, store

# lc0 takes ~1.5s to start (weights load + Metal init) against Stockfish's 50ms,
# so a worker handles a CHUNK of games and opens each net once for the whole
# chunk. Per-game engines would spend more time starting up than predicting.
# Engines are still quit inside the task, which is what keeps the pool from
# deadlocking the way the Stockfish backfill first did.
CHUNK_SIZE = 40


def weights_sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _predict_chunk(args: tuple[list[tuple[str, list[tuple[int, str]]]], str, dict[int, str], int]):
    """Worker: predict every position in a chunk of games, for every rating net."""
    games, binary, weights, nodes = args
    engines: dict[int, chess.engine.SimpleEngine] = {}
    out: list[tuple[str, int, int, str]] = []
    try:
        for rating, path in weights.items():
            engine = chess.engine.SimpleEngine.popen_uci([binary, f"--weights={path}"])
            engines[rating] = engine

        limit = chess.engine.Limit(nodes=nodes)
        for game_id, positions in games:
            for ply, fen in positions:
                board = chess.Board(fen)
                for rating, engine in engines.items():
                    move = engine.play(board, limit).move
                    out.append((game_id, ply, rating, move.uci()))
    finally:
        for engine in engines.values():
            try:
                engine.quit()
            except Exception:
                pass
    return out


def pending_games(conn: sqlite3.Connection, ratings: Sequence[int]) -> list[tuple[str, list]]:
    """Cohort games whose player moves have no predictions yet, with their positions."""
    lo, hi = store.cohort_bounds()
    rows = conn.execute(
        f"SELECT g.game_id FROM games g WHERE {store.COHORT_WHERE} "
        "AND NOT EXISTS (SELECT 1 FROM maia_moves m WHERE m.game_id = g.game_id "
        "                AND m.rating = ?) ORDER BY g.game_index",
        (lo, hi, ratings[0]),
    ).fetchall()
    out = []
    for row in rows:
        positions = conn.execute(
            "SELECT ply, fen_before FROM moves WHERE game_id = ? AND is_player = 1 ORDER BY ply",
            (row["game_id"],),
        ).fetchall()
        if positions:
            out.append((row["game_id"], [(p["ply"], p["fen_before"]) for p in positions]))
    return out


def run(
    conn: sqlite3.Connection,
    ratings: Sequence[int] | None = None,
    workers: int | None = None,
    on_progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    settings = config.MAIA
    ratings = list(ratings or settings.compare)
    weights = {r: settings.weights[r] for r in ratings}
    for rating, path in weights.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Maia weights missing for {rating}: {path}")
    shas = {r: weights_sha(p) for r, p in weights.items()}

    todo = pending_games(conn, ratings)
    if not todo:
        return {"games": 0, "predictions": 0, "elapsed_s": 0.0, "ratings": ratings}

    chunks = [todo[i:i + CHUNK_SIZE] for i in range(0, len(todo), CHUNK_SIZE)]
    workers = workers or os.cpu_count() or 4

    # The played move, for scoring matches without a second query.
    played = {
        (r["game_id"], r["ply"]): r["uci"]
        for r in conn.execute(
            "SELECT game_id, ply, uci FROM moves WHERE is_player = 1"
        )
    }

    started = time.perf_counter()
    done_games = predictions = 0
    now = store.utcnow()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_predict_chunk, (chunk, settings.lc0_binary, weights, settings.nodes)): chunk
            for chunk in chunks
        }
        for future in as_completed(futures):
            results = future.result()
            rows = [
                (gid, ply, rating, uci,
                 int(played.get((gid, ply)) == uci), shas[rating], now)
                for gid, ply, rating, uci in results
            ]
            with conn:
                conn.executemany(
                    "INSERT OR REPLACE INTO maia_moves "
                    "(game_id, ply, rating, predicted_uci, matched, weights_sha, computed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            predictions += len(rows)
            done_games += len(futures[future])
            if on_progress:
                on_progress(done_games, len(todo), time.perf_counter() - started)

    elapsed = time.perf_counter() - started
    return {
        "games": done_games,
        "predictions": predictions,
        "elapsed_s": elapsed,
        "ratings": ratings,
        "weights_sha": shas,
    }


def match_rates(conn: sqlite3.Connection, window: int = 100) -> list[dict[str, Any]]:
    """Match rate per rating net over the first and last `window` games."""
    ordered = [r["game_id"] for r in conn.execute(
        "SELECT DISTINCT g.game_id FROM maia_moves m JOIN games g USING (game_id) "
        "ORDER BY g.game_index"
    )]
    if not ordered:
        return []
    first_ids, last_ids = ordered[:window], ordered[-window:]

    def rate(ids: list[str], rating: int) -> tuple[float | None, int]:
        marks = ",".join("?" * len(ids))
        row = conn.execute(
            f"SELECT AVG(matched) a, COUNT(*) n FROM maia_moves "
            f"WHERE rating = ? AND game_id IN ({marks})", [rating] + ids
        ).fetchone()
        return (100 * row["a"] if row["a"] is not None else None, int(row["n"]))

    out = []
    for row in conn.execute("SELECT DISTINCT rating FROM maia_moves ORDER BY rating"):
        rating = row["rating"]
        f_val, f_n = rate(first_ids, rating)
        l_val, l_n = rate(last_ids, rating)
        o_val, o_n = rate(ordered, rating)
        out.append({"rating": rating, "first": f_val, "last": l_val, "overall": o_val,
                    "first_n": f_n, "last_n": l_n, "total_n": o_n})
    return out


def discriminating_rates(conn: sqlite3.Connection, low: int = 1200, high: int = 1500,
                         window: int = 100) -> dict[str, Any]:
    """The only Maia comparison that carries information: the disagreement subset.

    The two nets predict the same move on ~3 positions in 4. Those are obvious
    moves that every rating band plays, and they dominate the raw match rate --
    which is why maia-1200 and maia-1500 match rates track each other almost
    exactly and neither tells you much.

    Restricting to positions where the nets DISAGREE isolates the moves that
    actually separate rating bands, and asks: when the two levels would play
    different moves, which one did you play?
    """
    rows = conn.execute(
        "SELECT g.game_index, g.game_id, SUM(a.matched) low_m, SUM(b.matched) high_m, "
        "COUNT(*) n FROM maia_moves a JOIN maia_moves b USING (game_id, ply) "
        "JOIN games g USING (game_id) "
        "WHERE a.rating = ? AND b.rating = ? AND a.predicted_uci != b.predicted_uci "
        "GROUP BY g.game_id ORDER BY g.game_index",
        (low, high),
    ).fetchall()
    if not rows:
        return {}

    def summarise(subset) -> dict[str, Any]:
        lo_sum = sum(r["low_m"] for r in subset)
        hi_sum = sum(r["high_m"] for r in subset)
        decided = lo_sum + hi_sum
        return {
            "played_low": lo_sum,
            "played_high": hi_sum,
            "neither": sum(r["n"] for r in subset) - decided,
            "disagreements": sum(r["n"] for r in subset),
            # Of the times you played one of the two candidate moves, how often
            # was it the stronger band's move?
            "high_share": (100.0 * hi_sum / decided) if decided else None,
        }

    return {
        "low": low,
        "high": high,
        "first": summarise(rows[:window]),
        "last": summarise(rows[-window:]),
        "overall": summarise(rows),
    }


def render_discriminating(d: dict[str, Any], window: int = 100) -> str:
    if not d:
        return "No Maia disagreement data."
    lines = [
        f"Maia discriminating positions: only where maia-{d['low']} and "
        f"maia-{d['high']} would play DIFFERENT moves.",
        "Of the times you played one of the two candidates, how often was it the "
        f"{d['high']} move? NOT significance-tested.",
        "",
        f"{'window':>16} {'played '+str(d['low']):>12} {'played '+str(d['high']):>12} "
        f"{str(d['high'])+' share':>12}  disagreements",
    ]
    for label, key in ((f"first {window}", "first"), (f"last {window}", "last"),
                       ("all games", "overall")):
        s = d[key]
        share = "--" if s["high_share"] is None else f"{s['high_share']:.1f}%"
        lines.append(f"{label:>16} {s['played_low']:>12} {s['played_high']:>12} "
                     f"{share:>12}  {s['disagreements']}")
    return "\n".join(lines)


def render(rates: list[dict[str, Any]], window: int = 100) -> str:
    lines = [
        f"Maia move-matching: how often you played what each rating band plays.",
        f"First {window} games vs last {window}. NOT significance-tested.",
        "",
        f"{'net':>10} {'first':>8} {'last':>8} {'overall':>8}   sample (moves)",
    ]
    for r in rates:
        fmt = lambda v: "--" if v is None else f"{v:.1f}%"
        lines.append(
            f"{'maia-'+str(r['rating']):>10} {fmt(r['first']):>8} {fmt(r['last']):>8} "
            f"{fmt(r['overall']):>8}   {r['first_n']}/{r['last_n']} of {r['total_n']}"
        )
    return "\n".join(lines)
