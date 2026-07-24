"""Command line entry point: `chess-agent <command>`."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from . import config, ingest, openings, store, verify

OPENINGS_REPO = "https://raw.githubusercontent.com/lichess-org/chess-openings"


def cmd_init(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    store.init_db(conn)
    print(f"store ready at {args.db or config.DB_PATH} (schema v{config.SCHEMA_VERSION})")
    return 0


def cmd_vendor_openings(args: argparse.Namespace) -> int:
    """Re-vendor the opening book at a pinned commit."""
    directory = Path(args.dir or config.OPENINGS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    sha = args.sha
    for name in openings.TSV_FILES:
        url = f"{OPENINGS_REPO}/{sha}/{name}"
        with urllib.request.urlopen(url) as response:
            (directory / name).write_bytes(response.read())
        print(f"  {name}")
    (directory / "COMMIT_SHA").write_text(sha + "\n")
    positions = openings.book_positions(str(directory))
    print(f"vendored lichess-org/chess-openings @ {sha[:8]}: {len(positions)} book positions")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)

    def progress(added: int, skipped: int, failed: int) -> None:
        total = added + skipped + failed
        if total % 25 == 0:
            print(f"  ...{total} games", end="\r", flush=True)

    result = ingest.ingest(
        conn,
        username=args.username,
        max_games=args.max,
        full=args.full,
        on_progress=progress,
    )
    print(f"\ningested for {result['username']}: "
          f"{result['added']} new, {result['skipped']} skipped, {result['failed']} failed")
    for failure in result["failures"]:
        print(f"  ! {failure}")
    print(f"store now holds {result['total_games']} rapid games")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Time several engine settings on a small sample. Writes to evals_scratch only."""
    from dataclasses import replace
    from . import backfill

    conn = store.connect(args.db)
    lo, hi = store.cohort_bounds()
    rows = conn.execute(
        f"SELECT game_id FROM games WHERE {store.COHORT_WHERE} ORDER BY game_index LIMIT ?",
        (lo, hi, args.games),
    ).fetchall()
    games = [(r["game_id"], [m["uci"] for m in store.get_moves(conn, r["game_id"])])
             for r in rows]
    total_moves = conn.execute(
        f"SELECT SUM(n_plies) FROM games WHERE {store.COHORT_WHERE}", (lo, hi)).fetchone()[0]

    print(f"sample: {len(games)} games, {sum(len(u) for _, u in games)} plies")
    print(f"engine: {backfill.engine_version(config.STOCKFISH.binary)}")
    print(f"cohort to backfill: {total_moves:,} moves\n")
    print(f"{'setting':>15} {'wall_s':>8} {'pos/s':>8} {'mean_depth':>11} {'full_run':>10}")

    candidates = [
        ("movetime 50ms", replace(config.STOCKFISH, movetime_ms=50, depth=None)),
        ("movetime 100ms", replace(config.STOCKFISH, movetime_ms=100, depth=None)),
        ("movetime 200ms", replace(config.STOCKFISH, movetime_ms=200, depth=None)),
        ("depth 12", replace(config.STOCKFISH, movetime_ms=None, depth=12)),
        ("depth 14", replace(config.STOCKFISH, movetime_ms=None, depth=14)),
        ("depth 16", replace(config.STOCKFISH, movetime_ms=None, depth=16)),
    ]
    for label, settings in candidates:
        engine_id = backfill.register(conn, settings)
        result = backfill.run(conn, settings, games=games, table="evals_scratch",
                              engine_id=engine_id, run_label=label)
        depth = conn.execute(
            "SELECT AVG(depth_reached) FROM evals_scratch WHERE run_label = ?", (label,)
        ).fetchone()[0]
        est = total_moves / result["positions_per_s"] / 60
        print(f"{label:>15} {result['elapsed_s']:8.1f} {result['positions_per_s']:8.1f} "
              f"{depth:11.1f} {est:7.1f} min")

    print("\nBenchmark rows live in evals_scratch and are never read by metrics.")
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    from . import backfill

    conn = store.connect(args.db)
    settings = config.STOCKFISH
    print(f"engine:   {backfill.engine_version(settings.binary)}")
    print(f"settings: movetime={settings.movetime_ms}ms depth={settings.depth} "
          f"threads={settings.threads} workers={settings.workers or 'auto'}")

    engine_id = backfill.register(conn, settings)
    todo = backfill.pending_games(conn, engine_id)
    print(f"engine_id: {engine_id}\npending:   {len(todo)} games\n")
    if not todo:
        print("nothing to do -- every cohort game already has evals for this config")
        return 0

    def progress(done: int, total: int, elapsed: float) -> None:
        rate = done / elapsed if elapsed else 0
        eta = (total - done) / rate / 60 if rate else 0
        print(f"  {done}/{total} games  {elapsed/60:.1f} min elapsed  ~{eta:.1f} min left",
              end="\r", flush=True)

    result = backfill.run(conn, settings, engine_id=engine_id, on_progress=progress)
    print(f"\n\ndone: {result['games']} games, {result['positions']:,} positions "
          f"in {result['elapsed_s']/60:.1f} min ({result['positions_per_s']:.0f} pos/s)")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    report = verify.verify_store(conn, limit=args.limit)
    print(report.render())
    return 0 if report.ok else 1


def cmd_validate(args: argparse.Namespace) -> int:
    from . import validate

    conn = store.connect(args.db)
    report = validate.validate(conn, engine_id=args.engine_id)
    print(report.render())
    return 0 if report.passed else 1


def cmd_summary(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    meta = store.get_meta(conn)
    info = store.summary(conn)
    print(f"user: {meta.get('username')}  perf: {meta.get('perf_type')}")
    print(f"opening book: {(meta.get('openings_sha') or '?')[:8]}")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print("\nanalysis cohort (what metrics will actually see):")
    for key, value in store.cohort_summary(conn).items():
        print(f"  {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chess-agent", description=__doc__)
    parser.add_argument("--db", help="path to the SQLite store (default: data/chess_agent.sqlite)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("init", help="create the store").set_defaults(func=cmd_init)

    p_vendor = subparsers.add_parser("vendor-openings", help="download the opening book at a pinned commit")
    p_vendor.add_argument("--sha", required=True, help="lichess-org/chess-openings commit sha")
    p_vendor.add_argument("--dir", help="target directory")
    p_vendor.set_defaults(func=cmd_vendor_openings)

    p_ingest = subparsers.add_parser("ingest", help="stream rapid games from Lichess")
    p_ingest.add_argument("--username", help="defaults to the account behind LICHESS_TOKEN")
    p_ingest.add_argument("--max", type=int, help="cap the number of games (for smoke tests)")
    p_ingest.add_argument("--full", action="store_true", help="re-stream and re-parse everything")
    p_ingest.set_defaults(func=cmd_ingest)

    p_bench = subparsers.add_parser("benchmark", help="time engine settings on this machine")
    p_bench.add_argument("--games", type=int, default=20, help="sample size (default 20)")
    p_bench.set_defaults(func=cmd_benchmark)

    subparsers.add_parser(
        "backfill", help="run Stockfish over every cohort move"
    ).set_defaults(func=cmd_backfill)

    p_verify = subparsers.add_parser("verify", help="run the phase-1 reconstruction gate")
    p_verify.add_argument("--limit", type=int, help="check only the first N games")
    p_verify.set_defaults(func=cmd_verify)

    p_val = subparsers.add_parser("validate", help="phase-2 gate: agree with Lichess?")
    p_val.add_argument("--engine-id", help="defaults to the most recent local config")
    p_val.set_defaults(func=cmd_validate)

    subparsers.add_parser("summary", help="what is in the store").set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
