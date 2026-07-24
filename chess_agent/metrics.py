"""Per-game metrics, derived from `moves` + `evals`.

Counts the analysed player's moves only. Everything here is descriptive: this
layer answers "what were the numbers", never "did he improve". The MVP ships
without significance testing, so every rate is stored beside its denominator --
a rate without its sample size is how a 30-move endgame finding ends up reading
like a 700-game one.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from . import config, evals, store

# A player is "in time trouble" below this much clock. Rapid on Lichess is
# almost always 10+0, so one minute left is the last ~10% of the game clock.
TIME_TROUBLE_CENTIS = 6_000

# "Clearly winning" for the conversion metric, in win probability.
WINNING_WIN_PCT = 80.0

# Per-move centipawn loss is capped before averaging. Without a cap a single
# catastrophe in one game swamps the average for that game.
ACPL_CAP = 1_000.0

PHASES = ("opening", "middlegame", "endgame")


@dataclass
class GameMetrics:
    game_id: str
    engine_id: str
    moves: int = 0
    blunders: int = 0
    mistakes: int = 0
    inaccuracies: int = 0
    acpl: float | None = None
    acpl_moves: int = 0
    opening_moves: int = 0
    opening_blunders: int = 0
    middlegame_moves: int = 0
    middlegame_blunders: int = 0
    endgame_moves: int = 0
    endgame_blunders: int = 0
    time_trouble_moves: int = 0
    time_trouble_blunders: int = 0
    first_out_of_book_ply: int | None = None
    reached_winning: int = 0
    converted: int = 0


def compute_game(
    move_rows: list[sqlite3.Row],
    eval_rows: list[sqlite3.Row],
    initial_cp: int | None,
    game_id: str,
    engine_id: str,
    player_result: float | None,
) -> GameMetrics:
    judged = evals.judge_game(eval_rows, initial_cp)
    metrics = GameMetrics(game_id=game_id, engine_id=engine_id)

    cp_losses: list[float] = []
    for move in move_rows:
        ply = move["ply"]

        # Book depth is a property of the game, not of the player's moves, so it
        # is computed over every ply before the is_player filter below.
        if not move["book"] and metrics.first_out_of_book_ply is None:
            metrics.first_out_of_book_ply = ply

        if not move["is_player"]:
            continue
        judgement = judged.get(ply)
        if judgement is None:
            continue  # terminal position, or an eval we do not have

        metrics.moves += 1
        name = judgement.judgment.name
        if name == "blunder":
            metrics.blunders += 1
        elif name == "mistake":
            metrics.mistakes += 1
        elif name == "inaccuracy":
            metrics.inaccuracies += 1

        phase = move["phase"]
        setattr(metrics, f"{phase}_moves", getattr(metrics, f"{phase}_moves") + 1)
        if judgement.is_blunder:
            setattr(metrics, f"{phase}_blunders",
                    getattr(metrics, f"{phase}_blunders") + 1)

        clock = move["clock_centis"]
        if clock is not None and clock < TIME_TROUBLE_CENTIS:
            metrics.time_trouble_moves += 1
            if judgement.is_blunder:
                metrics.time_trouble_blunders += 1

        # Capped ACPL, decided positions excluded: swings in a +900 position are
        # engine noise, not skill, and would drown the signal from real play.
        if judgement.cp_loss is not None and not judgement.decided:
            cp_losses.append(min(judgement.cp_loss, ACPL_CAP))

        # Conversion: did the player ever stand clearly winning?
        mover_win_pct = (judgement.win_pct_after if judgement.mover_is_white
                         else 100 - judgement.win_pct_after)
        if mover_win_pct >= WINNING_WIN_PCT:
            metrics.reached_winning = 1

    metrics.acpl_moves = len(cp_losses)
    metrics.acpl = sum(cp_losses) / len(cp_losses) if cp_losses else None
    if metrics.reached_winning and player_result == 1.0:
        metrics.converted = 1
    return metrics


_COLUMNS = tuple(GameMetrics.__dataclass_fields__) + ("computed_at",)


def compute_all(conn: sqlite3.Connection, engine_id: str | None = None) -> dict[str, Any]:
    """Recompute metrics for every cohort game. Idempotent; safe to re-run."""
    from . import validate

    engine_id = engine_id or validate.resolve_engine_id(conn)
    initial_cp = conn.execute(
        "SELECT initial_cp FROM engine_configs WHERE engine_id = ?", (engine_id,)
    ).fetchone()["initial_cp"]

    games = store.analysis_games(conn)
    rows: list[tuple] = []
    now = store.utcnow()
    for game in games:
        gid = game["game_id"]
        eval_rows = conn.execute(
            "SELECT ply, cp, mate FROM evals WHERE game_id = ? AND engine_id = ? ORDER BY ply",
            (gid, engine_id),
        ).fetchall()
        if not eval_rows:
            continue
        metrics = compute_game(
            store.get_moves(conn, gid), eval_rows, initial_cp, gid, engine_id,
            game["player_result"],
        )
        record = asdict(metrics)
        record["computed_at"] = now
        rows.append(tuple(record[c] for c in _COLUMNS))

    with conn:
        conn.execute("DELETE FROM game_metrics WHERE engine_id = ?", (engine_id,))
        conn.executemany(
            f"INSERT INTO game_metrics ({', '.join(_COLUMNS)}) "
            f"VALUES ({', '.join('?' for _ in _COLUMNS)})",
            rows,
        )
    return {"engine_id": engine_id, "games": len(rows)}


# ---------------------------------------------------------------------------
# Descriptive aggregation -- what the narrator will read
# ---------------------------------------------------------------------------

@dataclass
class Descriptive:
    """One metric summarised over the cohort. No claim of significance."""

    metric: str
    phase: str | None
    first_value: float | None    # over the earliest `window` games
    last_value: float | None     # over the latest `window` games
    overall: float | None
    first_n: int                 # denominators travel with the numbers, always
    last_n: int
    total_n: int
    unit: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _rate(conn, numerator, denominator, where="") -> Any:
    return conn.execute(
        f"SELECT SUM({numerator}) num, SUM({denominator}) den FROM game_metrics "
        f"JOIN games USING (game_id) WHERE 1=1 {where} ORDER BY game_index"
    ).fetchone()


def describe(conn: sqlite3.Connection, window: int = 100) -> list[Descriptive]:
    """Summarise each metric over the first and last `window` games.

    Deliberately NOT a significance test. These are descriptions of what the
    numbers did; whether a difference is bigger than noise is not answered here
    and must not be implied by whoever narrates them.
    """
    ordered = [r["game_id"] for r in conn.execute(
        "SELECT g.game_id FROM game_metrics m JOIN games g USING (game_id) "
        "ORDER BY g.game_index"
    )]
    if not ordered:
        return []
    first_ids, last_ids = ordered[:window], ordered[-window:]

    def rate(ids: list[str], num: str, den: str) -> tuple[float | None, int]:
        marks = ",".join("?" * len(ids))
        row = conn.execute(
            f"SELECT SUM({num}) n, SUM({den}) d FROM game_metrics "
            f"WHERE game_id IN ({marks})", ids
        ).fetchone()
        if not row["d"]:
            return None, 0
        return 100.0 * row["n"] / row["d"], int(row["d"])

    def mean(ids: list[str], col: str) -> tuple[float | None, int]:
        marks = ",".join("?" * len(ids))
        row = conn.execute(
            f"SELECT AVG({col}) a, COUNT({col}) c FROM game_metrics "
            f"WHERE game_id IN ({marks})", ids
        ).fetchone()
        return (row["a"], int(row["c"] or 0))

    out: list[Descriptive] = []

    specs = [("blunder rate", None, "blunders", "moves")] + [
        (f"blunder rate", phase, f"{phase}_blunders", f"{phase}_moves") for phase in PHASES
    ] + [("blunder rate", "time trouble", "time_trouble_blunders", "time_trouble_moves"),
         ("conversion rate", None, "converted", "reached_winning")]

    for name, phase, num, den in specs:
        f_val, f_n = rate(first_ids, num, den)
        l_val, l_n = rate(last_ids, num, den)
        o_val, o_n = rate(ordered, num, den)
        out.append(Descriptive(name, phase, f_val, l_val, o_val, f_n, l_n, o_n, "%"))

    for name, col, unit in [("average centipawn loss", "acpl", "cp"),
                            ("book depth", "first_out_of_book_ply", "plies")]:
        f_val, f_n = mean(first_ids, col)
        l_val, l_n = mean(last_ids, col)
        o_val, o_n = mean(ordered, col)
        out.append(Descriptive(name, None, f_val, l_val, o_val, f_n, l_n, o_n, unit))

    return out


def render(records: Iterable[Descriptive], window: int = 100) -> str:
    lines = [
        f"Descriptive metrics: first {window} games vs last {window} games.",
        "NOT significance-tested -- differences here may be noise.",
        "",
        f"{'metric':>24} {'phase':>13} {'first':>9} {'last':>9} {'overall':>9}  sample",
    ]
    for r in records:
        fmt = lambda v: "--" if v is None else (f"{v:.1f}" + ("%" if r.unit == "%" else ""))
        lines.append(
            f"{r.metric:>24} {(r.phase or '-'):>13} {fmt(r.first_value):>9} "
            f"{fmt(r.last_value):>9} {fmt(r.overall):>9}  "
            f"n={r.first_n}/{r.last_n} of {r.total_n} {r.unit if r.unit != '%' else 'moves'}"
        )
    return "\n".join(lines)
