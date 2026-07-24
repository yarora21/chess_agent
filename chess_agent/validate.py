"""Phase-2 gate: do our blunder counts agree with Lichess's?

This does NOT test Stockfish. Lichess's server analysis is Stockfish too. What
it tests is our glue: the win% conversion, the classification thresholds, and
above all the point-of-view sign. A flipped sign would score good moves as
blunders and produce a confident, entirely backwards report -- and nothing
downstream would notice, because nothing downstream ever looks at a position.

Agreed tolerance, fixed before implementation: within 10%.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from . import config, evals, store

TOLERANCE = 0.10


@dataclass
class Validation:
    games: int = 0
    plies_compared: int = 0
    ours: int = 0
    lichess: int = 0
    agree: int = 0
    we_missed: int = 0
    we_extra: int = 0
    per_game: list[tuple[str, int, int]] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return self.ours / self.lichess if self.lichess else float("nan")

    @property
    def passed(self) -> bool:
        return abs(self.ratio - 1.0) <= TOLERANCE

    def render(self) -> str:
        pct = 100 * (self.ratio - 1)
        lines = [
            f"Compared {self.plies_compared:,} plies across {self.games} games that "
            f"carry both our evals and Lichess server analysis.",
            "",
            f"  blunders we found      {self.ours:,}",
            f"  blunders Lichess found {self.lichess:,}",
            f"  difference             {pct:+.1f}%   (tolerance +/-{TOLERANCE*100:.0f}%)",
            "",
            f"  we agree on            {self.agree:,}",
            f"  Lichess flagged, we did not   {self.we_missed:,}",
            f"  we flagged, Lichess did not   {self.we_extra:,}",
            "",
            f"VERDICT: {'PASS' if self.passed else 'FAIL'}",
        ]
        if not self.passed:
            worst = sorted(self.per_game, key=lambda t: -abs(t[1] - t[2]))[:5]
            lines += ["", "worst-disagreeing games (ours vs lichess):"]
            lines += [f"  {g}: {a} vs {b}" for g, a, b in worst]
        return "\n".join(lines)


def _lichess_blunders(analysis: list[dict[str, Any]]) -> set[int]:
    return {
        i for i, e in enumerate(analysis)
        if e.get("judgment", {}).get("name") == "Blunder"
    }


def _our_blunders(rows: list[sqlite3.Row], initial_cp: int | None) -> set[int]:
    """Classify every ply from our own evals.

    `rows` are evals ordered by ply, each holding the eval of the position AFTER
    that ply. The 'before' side of ply i is therefore row i-1, and for ply 0 it
    is the engine's eval of the starting position.
    """
    by_ply = {r["ply"]: r for r in rows}
    found: set[int] = set()
    for ply, row in by_ply.items():
        if ply == 0:
            cp_before, mate_before = initial_cp, None
        else:
            prev = by_ply.get(ply - 1)
            if prev is None:
                continue
            cp_before, mate_before = prev["cp"], prev["mate"]
        if cp_before is None and mate_before is None:
            continue
        judgment = evals.judge_move(
            cp_before=cp_before,
            mate_before=mate_before,
            cp_after=row["cp"],
            mate_after=row["mate"],
            mover_is_white=(ply % 2 == 0),
        )
        if judgment.is_blunder:
            found.add(ply)
    return found


def resolve_engine_id(conn: sqlite3.Connection) -> str:
    """The one engine config the `evals` table actually holds.

    Resolved from the data rather than from whichever config was registered most
    recently -- registering a config is free and the benchmark registers six of
    them, so "most recent" is not "the one we backfilled".

    Finding more than one config with evals is a principle-3 violation: a metrics
    run must read from exactly one local config, because eval-quality differences
    between configs would correlate with whatever order the games were analysed
    in and inject a trend that is not in the play.
    """
    rows = conn.execute(
        "SELECT engine_id, COUNT(*) n FROM evals GROUP BY engine_id ORDER BY n DESC"
    ).fetchall()
    if not rows:
        raise RuntimeError("`evals` is empty -- run the backfill first")
    if len(rows) > 1:
        detail = ", ".join(f"{r['engine_id']}={r['n']:,} rows" for r in rows)
        raise RuntimeError(
            f"evals holds {len(rows)} engine configs ({detail}). Metrics must read "
            "from exactly one; re-backfill or delete the stale config's rows."
        )
    return rows[0]["engine_id"]


def validate(conn: sqlite3.Connection, engine_id: str | None = None) -> Validation:
    if engine_id is None:
        engine_id = resolve_engine_id(conn)

    initial_cp = conn.execute(
        "SELECT initial_cp FROM engine_configs WHERE engine_id = ?", (engine_id,)
    ).fetchone()["initial_cp"]

    lo, hi = store.cohort_bounds()
    games = conn.execute(
        f"SELECT game_id, raw_json FROM games WHERE {store.COHORT_WHERE} "
        "AND has_lichess_analysis = 1 ORDER BY game_index",
        (lo, hi),
    ).fetchall()

    report = Validation()
    for game in games:
        rows = conn.execute(
            "SELECT ply, cp, mate FROM evals WHERE game_id = ? AND engine_id = ? ORDER BY ply",
            (game["game_id"], engine_id),
        ).fetchall()
        if not rows:
            continue
        analysis = json.loads(game["raw_json"])["analysis"]

        theirs = _lichess_blunders(analysis)
        ours = _our_blunders(rows, initial_cp)
        # Only compare plies both sides actually evaluated.
        comparable = min(len(rows), len(analysis))
        theirs = {p for p in theirs if p < comparable}
        ours = {p for p in ours if p < comparable}

        report.games += 1
        report.plies_compared += comparable
        report.ours += len(ours)
        report.lichess += len(theirs)
        report.agree += len(ours & theirs)
        report.we_missed += len(theirs - ours)
        report.we_extra += len(ours - theirs)
        report.per_game.append((game["game_id"], len(ours), len(theirs)))

    return report
