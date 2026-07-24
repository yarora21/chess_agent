-- Eval store schema. This file is the contract between pipeline stages
-- (ingest -> enrich -> metrics -> claims -> narrate). Changing it means adding a
-- migration note at the bottom of this file, in the same commit as the change.
--
-- Conventions that hold everywhere in this schema:
--   * ply is 0-based: ply 0 is White's first move. This matches the indexing of
--     the `clocks`, `analysis`, and `division` arrays/fields in the Lichess
--     export, so no off-by-one translation is ever needed at the boundary.
--   * timestamps from Lichess are integer milliseconds since epoch (as given);
--     timestamps we generate are ISO-8601 UTC strings.
--   * "player" always means the account being analyzed (meta.username);
--     "opponent" is the other side.
--
-- MIGRATIONS: see the bottom of this file.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- meta: single-row bookkeeping (id = 1) plus schema versioning.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS meta (
    id             INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL,
    username       TEXT,             -- Lichess username being analyzed (canonical case)
    perf_type      TEXT NOT NULL DEFAULT 'rapid',  -- rapid-only by design; never mix perf types
    openings_sha   TEXT,             -- pinned lichess-org/chess-openings commit used for book tags
    last_ingest_at TEXT
);

-- ---------------------------------------------------------------------------
-- games: one row per rapid game. Raw export payload is kept verbatim so that
-- re-parsing never requires re-fetching from the API.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_id        TEXT PRIMARY KEY,

    -- game_index is the trend regressor for the whole project: chronological
    -- position in the player's rapid history, 0-based, ordered by created_at.
    -- Rating is deliberately NOT the regressor (it is partly circular).
    -- Assigned/reassigned by store.reindex_games() after every ingest.
    game_index     INTEGER UNIQUE,

    rated          INTEGER NOT NULL,
    perf           TEXT NOT NULL,          -- must be 'rapid' (enforced at ingest)
    speed          TEXT,
    variant        TEXT,
    status         TEXT,
    winner         TEXT,                   -- 'white' | 'black' | NULL for draw
    created_at     INTEGER NOT NULL,       -- ms epoch
    last_move_at   INTEGER,

    -- Player-relative fields (denormalized for convenience; derived from white_*/black_*)
    player_color   TEXT NOT NULL,          -- 'white' | 'black'
    player_rating  INTEGER,
    player_rating_diff INTEGER,
    player_result  REAL,                   -- 1.0 win / 0.5 draw / 0.0 loss
    opponent_name  TEXT,
    opponent_rating INTEGER,
    opponent_rating_diff INTEGER,

    white_name     TEXT,
    white_rating   INTEGER,
    black_name     TEXT,
    black_rating   INTEGER,

    eco            TEXT,
    opening_name   TEXT,
    opening_ply    INTEGER,                -- Lichess's own opening-length marker

    clock_initial  INTEGER,                -- seconds
    clock_increment INTEGER,               -- seconds
    clock_total_moves INTEGER,

    -- Phase boundaries from the export's `division` field, in 0-based ply.
    -- NULL means the game never reached that phase.
    division_middlegame INTEGER,
    division_endgame    INTEGER,

    n_plies        INTEGER NOT NULL,       -- number of moves actually stored
    has_clocks     INTEGER NOT NULL DEFAULT 0,
    has_lichess_analysis INTEGER NOT NULL DEFAULT 0,

    moves_san      TEXT,                   -- space-separated SAN, as exported
    pgn            TEXT,
    raw_json       TEXT NOT NULL,          -- verbatim NDJSON record

    ingested_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_games_created_at ON games (created_at);
CREATE INDEX IF NOT EXISTS idx_games_eco ON games (eco);

-- ---------------------------------------------------------------------------
-- moves: one row per ply. Everything the metrics layer needs to slice a move
-- (phase, book status, whose move, time spent) without replaying the PGN.
-- fen_before is stored so the enrich stage can hand positions to Stockfish/Maia
-- and so board state is reconstructible without python-chess.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS moves (
    game_id        TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    ply            INTEGER NOT NULL,       -- 0-based
    move_number    INTEGER NOT NULL,       -- 1-based standard chess move number
    side           TEXT NOT NULL,          -- 'white' | 'black' — side that made this move
    is_player      INTEGER NOT NULL,       -- 1 if the analyzed player made this move

    san            TEXT NOT NULL,
    uci            TEXT NOT NULL,
    fen_before     TEXT NOT NULL,          -- position the move was played from
    epd_after      TEXT NOT NULL,          -- position after the move, EPD (no clocks) — book lookup key

    clock_centis   INTEGER,                -- clock remaining for `side` AFTER this move
    time_spent_centis INTEGER,             -- derived: previous clock - clock + increment

    phase          TEXT NOT NULL,          -- 'opening' | 'middlegame' | 'endgame'
    book           INTEGER NOT NULL,       -- 1 if epd_after is a known opening position
    book_eco       TEXT,                   -- ECO of the matched opening line, if book
    book_name      TEXT,

    PRIMARY KEY (game_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_moves_player_phase ON moves (is_player, phase);
CREATE INDEX IF NOT EXISTS idx_moves_game ON moves (game_id);

-- ---------------------------------------------------------------------------
-- engine_configs: engine identity. Principle 3 — evals are cached per
-- (game, engine-config), and a metrics run must read from exactly one local
-- config. Mixing configs within an analysis is an error: eval-quality
-- differences would correlate with analysis time and inject fake trends.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS engine_configs (
    engine_id      TEXT PRIMARY KEY,       -- stable hash of the identity fields below
    provenance     TEXT NOT NULL CHECK (provenance IN ('lichess', 'local')),
    engine_name    TEXT NOT NULL,          -- 'stockfish' | 'lc0-maia' | 'lichess-server'
    engine_version TEXT,
    depth          INTEGER,
    movetime_ms    INTEGER,
    threads        INTEGER,
    extra_json     TEXT,                   -- any further UCI options, canonical JSON
    created_at     TEXT NOT NULL
);

-- ---------------------------------------------------------------------------
-- evals: per-move engine evaluations, keyed by move AND engine identity.
-- Scores are from White's point of view, in centipawns, matching Lichess.
-- Exactly one of (cp, mate) is non-NULL.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evals (
    game_id        TEXT NOT NULL,
    ply            INTEGER NOT NULL,
    engine_id      TEXT NOT NULL REFERENCES engine_configs(engine_id),

    cp             INTEGER,                -- centipawns, White POV
    mate           INTEGER,                -- mate-in-N, White POV (negative = Black mates)
    best_move      TEXT,                   -- UCI
    pv             TEXT,
    depth_reached  INTEGER,
    nodes          INTEGER,

    computed_at    TEXT NOT NULL,

    PRIMARY KEY (game_id, ply, engine_id),
    FOREIGN KEY (game_id, ply) REFERENCES moves(game_id, ply) ON DELETE CASCADE,
    CHECK ((cp IS NULL) != (mate IS NULL))
);

CREATE INDEX IF NOT EXISTS idx_evals_engine ON evals (engine_id);

-- ---------------------------------------------------------------------------
-- evals_scratch: benchmark runs write here and are discarded. Keeping them out
-- of `evals` is what makes "analyze each game once per (game, engine-config)"
-- safe to enforce — a benchmark must never become part of the real backfill.
-- Same columns as `evals`, minus the foreign keys.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evals_scratch (
    game_id        TEXT NOT NULL,
    ply            INTEGER NOT NULL,
    engine_id      TEXT NOT NULL,
    cp             INTEGER,
    mate           INTEGER,
    best_move      TEXT,
    pv             TEXT,
    depth_reached  INTEGER,
    nodes          INTEGER,
    computed_at    TEXT NOT NULL,
    run_label      TEXT,                   -- which benchmark run this came from
    elapsed_ms     REAL,                   -- wall-clock for this position (benchmarking)
    PRIMARY KEY (game_id, ply, engine_id, run_label)
);

-- ---------------------------------------------------------------------------
-- MIGRATIONS
-- ---------------------------------------------------------------------------
-- v1 (phase 1): initial schema — meta, games, moves, engine_configs, evals,
--     evals_scratch. Created for the ingest + storage phase; the evals tables
--     are defined but left empty until phase 2.
