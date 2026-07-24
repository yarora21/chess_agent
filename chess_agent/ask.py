"""Read-only Q&A agent over the store.

The agent may retrieve and verbalize. It may never evaluate a position, classify
a move, or make a statistical judgment itself -- every chess fact it states must
come out of a tool call, and every claim about a game must cite that game's id.
Questions the store cannot answer get "the data doesn't cover that", not a
plausible-sounding guess.

The read-only guarantee is enforced by SQLite (`mode=ro`), not by the prompt.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import evals, maia, metrics, store

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You answer questions about one chess player's game history by querying a \
database of pre-computed statistics. You are a retrieval interface, not a chess \
engine and not a coach.

RULES:
1. Every factual claim must come from a tool result. Never state a chess fact \
you did not retrieve. Never compute a number the tools did not return.
2. Never evaluate a position, judge a move, or explain why a move was bad. The \
database already contains those judgments -- retrieve them. If asked "why was \
this a blunder", you may report the evaluation before and after; you may not \
explain the chess.
3. Cite game IDs whenever you reference specific games.
4. These numbers carry NO significance testing. Never say "significantly", \
"proves", or "you improved". Describe what the numbers are. When a sample is \
small, say so.
5. If the data cannot answer the question, say "the data doesn't cover that" \
and state what you would need. Do not improvise, extrapolate, or fall back on \
general chess knowledge. This is the most important rule: a confident wrong \
answer is far worse than an admitted gap.

Answer in a few sentences unless asked for more. Lead with the answer.
"""

# The agent's connection. Module-level so the tool functions -- which the SDK
# calls with only their declared arguments -- can reach it.
_conn: sqlite3.Connection | None = None


def _db() -> sqlite3.Connection:
    if _conn is None:
        raise RuntimeError("no connection bound; call ask() rather than the tools")
    return _conn


def _rows(cursor) -> list[dict[str, Any]]:
    return [dict(r) for r in cursor]


# ---------------------------------------------------------------------------
# Tools. Docstrings are the model's only documentation -- keep them precise.
# ---------------------------------------------------------------------------

def tool_overview() -> str:
    """Summarise what data exists: how many games, moves, the rating range, and
    which metrics are available. Call this first when unsure what can be asked."""
    info = store.cohort_summary(_db())
    names = sorted({d.metric for d in metrics.describe(_db())})
    return json.dumps({**info, "available_metrics": names,
                       "note": "no significance testing; descriptive only"})


def tool_metric_summary() -> str:
    """Every metric compared across the first 100 vs last 100 games, with sample
    sizes. Use for questions about how a rate changed over time."""
    return json.dumps([d.as_record() for d in metrics.describe(_db())])


def tool_maia_summary() -> str:
    """Maia move-matching: how often the player played the move a human of a
    given rating plays, plus the discriminating-position breakdown."""
    return json.dumps({
        "match_rates": maia.match_rates(_db()),
        "discriminating_positions": maia.discriminating_rates(_db()),
    })


def tool_find_games(order_by: str = "middlegame_blunders", descending: bool = True,
                    limit: int = 10, result: str | None = None) -> str:
    """Find games ranked by a metric column, returning their IDs.

    Args:
        order_by: a game_metrics column, e.g. blunders, middlegame_blunders,
            endgame_blunders, acpl, time_trouble_blunders, first_out_of_book_ply.
        descending: True for worst/highest first.
        limit: how many games (max 50).
        result: filter to 'win', 'loss', or 'draw'. None for all.
    """
    allowed = {c["name"] for c in _db().execute("PRAGMA table_info(game_metrics)")}
    if order_by not in allowed:
        return json.dumps({"error": f"unknown column {order_by!r}",
                           "available": sorted(allowed)})
    where = {"win": "AND g.player_result = 1.0", "loss": "AND g.player_result = 0.0",
             "draw": "AND g.player_result = 0.5"}.get(result or "", "")
    lo, hi = store.cohort_bounds()
    rows = _rows(_db().execute(
        f"SELECT g.game_id, g.game_index, g.player_rating, g.opponent_rating, "
        f"g.player_result, g.eco, g.opening_name, m.{order_by} AS value, m.moves "
        f"FROM game_metrics m JOIN games g USING (game_id) "
        f"WHERE g.game_index BETWEEN ? AND ? {where} "
        f"ORDER BY m.{order_by} {'DESC' if descending else 'ASC'} LIMIT ?",
        (lo, hi, min(int(limit), 50)),
    ))
    return json.dumps(rows)


def tool_game_detail(game_id: str) -> str:
    """Everything known about one game: metadata and its computed metrics."""
    game = _db().execute(
        "SELECT game_id, game_index, created_at, player_color, player_rating, "
        "opponent_rating, player_result, status, eco, opening_name, n_plies "
        "FROM games WHERE game_id = ?", (game_id,)).fetchone()
    if game is None:
        return json.dumps({"error": f"no game {game_id!r}"})
    m = _db().execute("SELECT * FROM game_metrics WHERE game_id = ?", (game_id,)).fetchone()
    return json.dumps({"game": dict(game), "metrics": dict(m) if m else None})


def tool_game_moves(game_id: str, only_mistakes: bool = True) -> str:
    """The player's moves in one game with their engine judgments.

    Args:
        game_id: which game.
        only_mistakes: True returns only blunders/mistakes/inaccuracies (the
            usual case); False returns every player move, which can be long.
    """
    from . import validate

    engine_id = validate.resolve_engine_id(_db())
    initial = _db().execute(
        "SELECT initial_cp FROM engine_configs WHERE engine_id = ?", (engine_id,)
    ).fetchone()["initial_cp"]
    eval_rows = _db().execute(
        "SELECT ply, cp, mate FROM evals WHERE game_id = ? AND engine_id = ? ORDER BY ply",
        (game_id, engine_id)).fetchall()
    if not eval_rows:
        return json.dumps({"error": f"no evals for {game_id!r}"})

    judged = evals.judge_game(eval_rows, initial)
    out = []
    for m in _db().execute(
            "SELECT ply, move_number, san, phase, clock_centis, is_player "
            "FROM moves WHERE game_id = ? AND is_player = 1 ORDER BY ply", (game_id,)):
        j = judged.get(m["ply"])
        if j is None or (only_mistakes and j.judgment.name == "ok"):
            continue
        out.append({"ply": m["ply"], "move_number": m["move_number"], "san": m["san"],
                    "phase": m["phase"], "judgment": j.judgment.name,
                    "win_pct_lost": round(j.judgment.win_pct_lost, 1),
                    "clock_centis": m["clock_centis"]})
    return json.dumps({"game_id": game_id, "moves": out, "count": len(out)})


def tool_opening_breakdown(limit: int = 15) -> str:
    """Results and blunder counts grouped by opening (ECO code)."""
    lo, hi = store.cohort_bounds()
    return json.dumps(_rows(_db().execute(
        "SELECT g.eco, g.opening_name, COUNT(*) games, "
        "ROUND(AVG(g.player_result), 3) score, SUM(m.blunders) blunders, "
        "SUM(m.moves) moves FROM game_metrics m JOIN games g USING (game_id) "
        "WHERE g.game_index BETWEEN ? AND ? GROUP BY g.eco "
        "ORDER BY games DESC LIMIT ?", (lo, hi, min(int(limit), 50)))))


TOOL_FUNCTIONS = {
    "tool_overview": tool_overview,
    "tool_metric_summary": tool_metric_summary,
    "tool_maia_summary": tool_maia_summary,
    "tool_find_games": tool_find_games,
    "tool_game_detail": tool_game_detail,
    "tool_game_moves": tool_game_moves,
    "tool_opening_breakdown": tool_opening_breakdown,
}


def ask(question: str, db_path: str | None = None, model: str = MODEL,
        verbose: bool = False) -> str:
    """Answer one question against the store, using read-only tools."""
    global _conn
    import anthropic
    from anthropic import beta_tool

    _conn = store.connect_readonly(db_path)
    try:
        tools = [beta_tool(fn) for fn in TOOL_FUNCTIONS.values()]
        client = anthropic.Anthropic()
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=[{"role": "user", "content": question}],
        )
        last = None
        for message in runner:
            last = message
            if verbose:
                for block in message.content:
                    if block.type == "tool_use":
                        print(f"  [tool] {block.name}({block.input})")
        if last is None:
            return "(no response)"
        return "".join(b.text for b in last.content if b.type == "text").strip()
    finally:
        _conn.close()
        _conn = None
