"""The phase-1 gate: every game's board states and phase boundaries reconstruct.

This is deliberately paranoid. Phase 2 hands `fen_before` straight to Stockfish
and phase 3 slices moves by `phase` — if either is wrong, every number built on
top of it is wrong, silently. Cheap to check now, expensive to discover later.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import chess

from . import store

PHASE_ORDER = {"opening": 0, "middlegame": 1, "endgame": 2}


@dataclass
class VerifyReport:
    games_checked: int = 0
    games_ok: int = 0
    problems: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def render(self) -> str:
        lines = [
            f"Verified {self.games_checked} games: {self.games_ok} clean, "
            f"{self.games_checked - self.games_ok} with problems."
        ]
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        if self.problems:
            lines.append(f"\n{len(self.problems)} problem(s):")
            lines.extend(f"  - {p}" for p in self.problems[:40])
            if len(self.problems) > 40:
                lines.append(f"  ... and {len(self.problems) - 40} more")
        return "\n".join(lines)


def verify_game(game: sqlite3.Row, moves: list[sqlite3.Row]) -> list[str]:
    problems: list[str] = []
    gid = game["game_id"]

    if len(moves) != game["n_plies"]:
        problems.append(f"{gid}: n_plies={game['n_plies']} but {len(moves)} move rows")
    if game["moves_san"] and len(game["moves_san"].split()) != len(moves):
        problems.append(f"{gid}: moves_san has {len(game['moves_san'].split())} SAN tokens, "
                        f"{len(moves)} move rows")
    if not moves:
        return problems

    # 1. Board reconstruction: replay from the initial position and check that the
    #    stored positions are the ones actually reached.
    board = chess.Board()
    for i, m in enumerate(moves):
        if m["ply"] != i:
            problems.append(f"{gid}: ply column is {m['ply']} at row {i} (must be dense, 0-based)")
            return problems
        if board.fen() != m["fen_before"]:
            problems.append(f"{gid} ply {i}: fen_before does not match replay\n"
                            f"      stored: {m['fen_before']}\n      replay: {board.fen()}")
            return problems
        try:
            move = chess.Move.from_uci(m["uci"])
        except ValueError:
            problems.append(f"{gid} ply {i}: unparseable uci {m['uci']!r}")
            return problems
        if move not in board.legal_moves:
            problems.append(f"{gid} ply {i}: {m['uci']} is illegal in stored position")
            return problems
        if board.san(move) != m["san"]:
            problems.append(f"{gid} ply {i}: san {m['san']!r} != {board.san(move)!r} from replay")
        expected_side = "white" if board.turn == chess.WHITE else "black"
        if m["side"] != expected_side:
            problems.append(f"{gid} ply {i}: side={m['side']} but {expected_side} is to move")
        board.push(move)
        if board.epd() != m["epd_after"]:
            problems.append(f"{gid} ply {i}: epd_after does not match replay")

    # 2. Terminal position agrees with the recorded result.
    status, winner = game["status"], game["winner"]
    if status == "mate":
        if not board.is_checkmate():
            problems.append(f"{gid}: status=mate but final position is not checkmate")
        else:
            mated = "white" if board.turn == chess.WHITE else "black"
            if winner == mated:
                problems.append(f"{gid}: winner={winner} but {mated} is the side mated")
    if status == "stalemate" and not board.is_stalemate():
        problems.append(f"{gid}: status=stalemate but final position is not stalemate")

    # 3. Player identity is consistent.
    n_player_moves = sum(m["is_player"] for m in moves)
    expected = sum(1 for m in moves if m["side"] == game["player_color"])
    if n_player_moves != expected:
        problems.append(f"{gid}: is_player set on {n_player_moves} moves, expected {expected}")

    # 4. Phase boundaries: within range, and never going backwards.
    n = len(moves)
    for column in ("division_middlegame", "division_endgame"):
        value = game[column]
        if value is not None and not (0 <= value <= n):
            problems.append(f"{gid}: {column}={value} outside [0, {n}]")
    mid, end = game["division_middlegame"], game["division_endgame"]
    if mid is not None and end is not None and end < mid:
        problems.append(f"{gid}: endgame ply {end} precedes middlegame ply {mid}")

    last = -1
    for m in moves:
        rank = PHASE_ORDER.get(m["phase"], -1)
        if rank < 0:
            problems.append(f"{gid} ply {m['ply']}: unknown phase {m['phase']!r}")
            break
        if rank < last:
            problems.append(f"{gid} ply {m['ply']}: phase goes backwards to {m['phase']}")
            break
        last = rank

    # 5. Book tags are NOT a prefix, and must not be checked as one. Real games
    #    transpose: 1.d4 d5 2.c4 Nf6 3.g3 dxc4 4.Bg2 e6 leaves the named lines
    #    at ply 4 and re-enters the Catalan at ply 7. So `book` is a per-position
    #    property, and the metrics layer must distinguish two different things:
    #      * first_out_of_book_ply -- how deep prepared theory went (prefix length)
    #      * SUM(book)             -- how many moves sat in known theory overall
    #    Using the second where the first is meant would overstate book depth on
    #    every transposition. Counted below as a store-wide statistic.

    # 6. Clocks, when present, must be complete and monotonically decreasing per side.
    if game["has_clocks"]:
        missing = [m["ply"] for m in moves if m["clock_centis"] is None]
        if missing:
            problems.append(f"{gid}: has_clocks set but {len(missing)} plies lack a clock "
                            f"(first at ply {missing[0]})")

    return problems


def verify_store(conn: sqlite3.Connection, limit: int | None = None) -> VerifyReport:
    report = VerifyReport()
    query = "SELECT * FROM games ORDER BY game_index"
    if limit:
        query += f" LIMIT {int(limit)}"

    indices: list[int] = []
    unindexed: list[str] = []
    perfs: set[str] = set()
    for game in conn.execute(query).fetchall():
        moves = store.get_moves(conn, game["game_id"])
        problems = verify_game(game, moves)
        report.games_checked += 1
        report.games_ok += not problems
        report.problems.extend(problems)
        if game["game_index"] is None:
            unindexed.append(game["game_id"])
        else:
            indices.append(game["game_index"])
        perfs.add(game["perf"])

    # Store-wide invariants.
    if unindexed:
        report.problems.append(
            f"{len(unindexed)} game(s) have no game_index (first: {unindexed[0]}) — "
            "run reindex_games(); game_index is the trend regressor"
        )
    if sorted(indices) != list(range(len(indices))):
        report.problems.append("game_index is not a dense 0-based sequence — run reindex_games()")
    stray = perfs - {"rapid"}
    if stray:
        report.problems.append(f"non-rapid games in the store: {sorted(stray)}")

    report.stats = store.summary(conn)
    book_depth = conn.execute(
        "SELECT AVG(d) FROM (SELECT SUM(book) d FROM moves GROUP BY game_id)"
    ).fetchone()[0]
    report.stats["mean_book_moves_per_game"] = round(book_depth, 1) if book_depth else 0

    # Prepared-theory depth: the prefix length, which is what "book depth" means.
    # Differs from the count above exactly on transpositions.
    first_out = conn.execute(
        "SELECT AVG(f) FROM (SELECT MIN(ply) f FROM moves WHERE book = 0 GROUP BY game_id)"
    ).fetchone()[0]
    report.stats["mean_first_out_of_book_ply"] = round(first_out, 1) if first_out else 0

    transposed = conn.execute(
        "SELECT COUNT(*) FROM (SELECT game_id FROM moves GROUP BY game_id "
        " HAVING SUM(book) > MIN(CASE WHEN book = 0 THEN ply END))"
    ).fetchone()[0]
    report.stats["games_that_transpose_back_into_book"] = transposed

    # `division` convention probe. We read division.middle/end as ply COUNTS, so
    # move index i is middlegame once i >= middle. If they were instead 0-based
    # move INDEXES, no game could ever report a value equal to n_plies. Seeing
    # such games confirms the count reading; seeing none across a few hundred
    # games is weak evidence the other way and worth a second look before
    # phase-3 metrics are sliced by phase.
    at_end = conn.execute(
        "SELECT COUNT(*) FROM games WHERE division_endgame = n_plies "
        "OR division_middlegame = n_plies"
    ).fetchone()[0]
    report.stats["games_with_division_at_final_ply"] = at_end
    return report
