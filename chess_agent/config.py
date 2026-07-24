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

SCHEMA_VERSION = 1

# Rapid only. Bullet/blitz/rapid have wildly different blunder profiles and
# rating histories; aggregating across them is a correctness bug, not a setting.
PERF_TYPE = "rapid"

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

    binary: str = os.environ.get("STOCKFISH_PATH", "stockfish")
    movetime_ms: int | None = 100
    depth: int | None = None  # set one of movetime_ms / depth, not both
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
    weights: dict[int, str] = field(
        default_factory=lambda: {
            1200: os.environ.get("MAIA_1200_WEIGHTS", "weights/maia-1200.pb.gz"),
            1500: os.environ.get("MAIA_1500_WEIGHTS", "weights/maia-1500.pb.gz"),
        }
    )
    nodes: int = 1


STOCKFISH = StockfishSettings()
MAIA = MaiaSettings()
