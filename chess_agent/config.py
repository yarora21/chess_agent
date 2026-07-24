"""Single source of truth for paths and engine settings.

Engine settings live here (and only here) so that benchmark results translate
directly into production settings: you benchmark, you edit one dataclass, you
re-run. Phase 1 does not read the engine settings at all — they are declared now
so phase 2 has nowhere else to put them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("CHESS_AGENT_DATA", REPO_ROOT / "data"))
DB_PATH = DATA_DIR / "chess_agent.sqlite"
OPENINGS_DIR = DATA_DIR / "openings"
SCHEMA_PATH = PACKAGE_ROOT / "schema.sql"

SCHEMA_VERSION = 3

# Rapid only. Bullet/blitz/rapid have wildly different blunder profiles and
# rating histories; aggregating across them is a correctness bug, not a setting.
PERF_TYPE = "rapid"

# ---------------------------------------------------------------------------
# Analysis cohort: which stored games the metrics/claims stages are allowed to
# see. Ingest keeps everything; analysis narrows.
#
# Games 0-1 are stale one-offs (2021-09 and 2023-02) sitting three years before
# the real run, which begins 2026-01-26.
# Games 2-43 are the provisional-rating settling period: Lichess started the
# account at a guessed 1500 and corrected downward to ~1140 while actual
# strength was roughly flat. Rating moves there reflect Lichess calibrating,
# not the player changing.
#
# CAVEAT, deliberately on the record: index 44 is the rating trough, and
# choosing a start point because the rating bottomed there selects on the
# outcome -- it makes any subsequent climb look bigger than it was. The trend
# fits regress on game index rather than rating, which blunts this, but any
# narration that quotes the rating range inherits the bias. A defensible
# alternative is to cut on Lichess's own provisional flag instead. Revisit
# before phase 5 narration quotes a rating span.
ANALYSIS_START_INDEX = 44
ANALYSIS_END_INDEX = 767

# Lichess API: one request at a time, and a full 60s wait on HTTP 429.
# Rate limits are unpublished and dynamic.
RATE_LIMIT_BACKOFF_S = 60
MAX_RATE_LIMIT_RETRIES = 5


@dataclass(frozen=True)
class StockfishSettings:
    """Phase 2 settings. Benchmark ~20 games on this machine before trusting these.

    Depth 12-14 or 0.1-0.3 s/move is sufficient at the 1200-1500 band; the errors
    that matter are 200+ cp. One single-threaded engine per physical core --
    never one engine with many threads (Lazy SMP time-to-depth scales poorly).
    """

    binary: str = os.environ.get("STOCKFISH_PATH") or (
        str(REPO_ROOT / "bin" / "stockfish")
        if (REPO_ROOT / "bin" / "stockfish").exists()
        else "stockfish"
    )
    # Set exactly one of movetime_ms / depth.
    #
    # FIXED DEPTH, not fixed movetime -- a correctness choice, not a speed one.
    # Fixed movetime makes eval quality a function of how busy the machine is.
    # Games are handed to the pool in game_index order, so a slow patch (another
    # program hogging cores) degrades a *contiguous run of game indexes*, which
    # is indistinguishable from a real trend in play quality. That is exactly the
    # artifact principle 3 exists to prevent, arriving through machine load.
    # Fixed depth is reproducible: same engine, same position, same answer,
    # whatever else the laptop is doing.
    #
    # Benchmarked on this machine (8 cores, Stockfish 18, 6-game sample):
    #   depth 12       5.4 min full run     depth 14      12.9 min
    #   depth 16      32.8 min              movetime 100ms 25.5 min
    # Depth 14 sits in the 12-14 band the research doc calls sufficient at the
    # 1200-1500 level, where the errors that matter are 200+ cp.
    movetime_ms: int | None = None
    depth: int | None = 14
    threads: int = 1
    hash_mb: int = 64
    # Optionally stop analyzing once eval passes this magnitude; capped ACPL
    # ignores those moves anyway. None disables the cutoff.
    abort_above_cp: int | None = 800
    workers: int | None = None  # None -> physical core count


@dataclass(frozen=True)
class MaiaSettings:
    """Phase 4 settings: Lc0 + original Maia weights, policy net at nodes=1 (no search)."""

    lc0_binary: str = os.environ.get("LC0_PATH", "lc0")
    # Absolute paths: lc0 resolves --weights against ITS cwd, and the backfill
    # runs engines from worker processes whose cwd you should not rely on.
    weights: dict[int, str] = field(
        default_factory=lambda: {
            rating: str(REPO_ROOT / "weights" / f"maia-{rating}.pb.gz")
            for rating in (1100, 1200, 1300, 1400, 1500, 1600)
        }
    )
    # nodes=1 queries the policy head directly with no search -- we want "what
    # would a human at this rating play", not "what is best".
    nodes: int = 1
    # The two nets the report contrasts. Others are downloaded and available.
    compare: tuple[int, int] = (1200, 1500)


STOCKFISH = StockfishSettings()
MAIA = MaiaSettings()
