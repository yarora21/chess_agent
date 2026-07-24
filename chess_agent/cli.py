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


def cmd_verify(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    report = verify.verify_store(conn, limit=args.limit)
    print(report.render())
    return 0 if report.ok else 1


def cmd_summary(args: argparse.Namespace) -> int:
    conn = store.connect(args.db)
    meta = store.get_meta(conn)
    info = store.summary(conn)
    print(f"user: {meta.get('username')}  perf: {meta.get('perf_type')}")
    print(f"opening book: {(meta.get('openings_sha') or '?')[:8]}")
    for key, value in info.items():
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

    p_verify = subparsers.add_parser("verify", help="run the phase-1 reconstruction gate")
    p_verify.add_argument("--limit", type=int, help="check only the first N games")
    p_verify.set_defaults(func=cmd_verify)

    subparsers.add_parser("summary", help="what is in the store").set_defaults(func=cmd_summary)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
