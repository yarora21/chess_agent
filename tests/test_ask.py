"""Phase 6: the read-only Q&A tools.

Tests exercise the tool functions directly against the real store -- no model
calls, so these are fast and deterministic. The important ones prove the
read-only guarantee is real and that bad input is refused rather than guessed at.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from chess_agent import ask, config, store

pytestmark = pytest.mark.skipif(
    not config.DB_PATH.exists(), reason="needs an ingested store"
)


@pytest.fixture(autouse=True)
def bound_connection():
    ask._conn = store.connect_readonly()
    yield
    ask._conn.close()
    ask._conn = None


# ---------------------------------------------------------------------------
# The read-only guarantee
# ---------------------------------------------------------------------------

def test_the_agents_connection_cannot_write():
    """An instruction not to write is not a control. SQLite must refuse."""
    with pytest.raises(sqlite3.OperationalError):
        ask._conn.execute("DELETE FROM games")


def test_the_agents_connection_cannot_create_tables():
    with pytest.raises(sqlite3.OperationalError):
        ask._conn.execute("CREATE TABLE evil (x INTEGER)")


def test_the_agents_connection_can_still_read():
    assert ask._conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] > 0


# ---------------------------------------------------------------------------
# Tools return usable, grounded data
# ---------------------------------------------------------------------------

def test_overview_reports_the_cohort_and_flags_the_missing_stats():
    data = json.loads(ask.tool_overview())
    assert data["cohort_games"] > 0
    assert data["available_metrics"]
    assert "no significance testing" in data["note"]


def test_metric_summary_carries_sample_sizes():
    records = json.loads(ask.tool_metric_summary())
    assert records
    for r in records:
        assert "first_n" in r and "total_n" in r, "denominators must reach the agent"


def test_find_games_returns_ids_that_can_be_looked_up():
    games = json.loads(ask.tool_find_games(order_by="blunders", limit=3))
    assert len(games) == 3
    detail = json.loads(ask.tool_game_detail(games[0]["game_id"]))
    assert detail["game"]["game_id"] == games[0]["game_id"]


def test_find_games_is_actually_ordered():
    games = json.loads(ask.tool_find_games(order_by="blunders", descending=True, limit=5))
    values = [g["value"] for g in games]
    assert values == sorted(values, reverse=True)


def test_find_games_can_filter_by_result():
    for result, expected in (("win", 1.0), ("loss", 0.0), ("draw", 0.5)):
        games = json.loads(ask.tool_find_games(result=result, limit=3))
        assert all(g["player_result"] == expected for g in games)


def test_find_games_stays_inside_the_analysis_cohort():
    lo, hi = store.cohort_bounds()
    games = json.loads(ask.tool_find_games(limit=50))
    assert all(lo <= g["game_index"] <= hi for g in games)


def test_game_moves_returns_judged_mistakes_only_by_default():
    gid = json.loads(ask.tool_find_games(order_by="blunders", limit=1))[0]["game_id"]
    data = json.loads(ask.tool_game_moves(gid))
    assert data["count"] > 0
    assert all(m["judgment"] != "ok" for m in data["moves"])
    assert all("san" in m and "win_pct_lost" in m for m in data["moves"])


def test_game_moves_can_return_everything():
    gid = json.loads(ask.tool_find_games(order_by="blunders", limit=1))[0]["game_id"]
    mistakes = json.loads(ask.tool_game_moves(gid, only_mistakes=True))["count"]
    everything = json.loads(ask.tool_game_moves(gid, only_mistakes=False))["count"]
    assert everything > mistakes


def test_opening_breakdown_groups_by_eco():
    rows = json.loads(ask.tool_opening_breakdown(limit=5))
    assert rows and all("eco" in r and "games" in r for r in rows)


def test_maia_summary_includes_the_discriminating_subset():
    data = json.loads(ask.tool_maia_summary())
    assert data["match_rates"]
    assert data["discriminating_positions"]["overall"]["disagreements"] > 0


# ---------------------------------------------------------------------------
# Bad input is refused, not guessed at
# ---------------------------------------------------------------------------

def test_an_unknown_column_is_refused_with_the_valid_options():
    data = json.loads(ask.tool_find_games(order_by="brilliance"))
    assert "error" in data and "available" in data


def test_a_sql_injection_attempt_in_order_by_is_rejected():
    data = json.loads(ask.tool_find_games(order_by="blunders; DROP TABLE games"))
    assert "error" in data, "order_by must be validated against real columns"
    assert ask._conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] > 0


def test_an_unknown_game_id_is_reported_not_invented():
    assert "error" in json.loads(ask.tool_game_detail("nonexistent"))
    assert "error" in json.loads(ask.tool_game_moves("nonexistent"))


def test_the_limit_is_capped():
    games = json.loads(ask.tool_find_games(limit=9999))
    assert len(games) <= 50


# ---------------------------------------------------------------------------
# Prompt contract
# ---------------------------------------------------------------------------

def test_the_prompt_forbids_improvising_and_requires_citations():
    prompt = ask.SYSTEM_PROMPT.lower()
    assert "the data doesn't cover that" in prompt
    assert "cite game ids" in prompt
    assert "never evaluate a position" in prompt
    assert "significance testing" in prompt
