"""Book / out-of-book tagging against the vendored lichess-org/chess-openings TSVs.

Why this source: it is the same data behind Lichess's own ECO tagging, so our
book depth agrees with the `opening` field in the export rather than drifting
from it. It is vendored at a pinned commit (data/openings/COMMIT_SHA) so the tag
is deterministic — the same game always gets the same book flags.

A move is `book` if the position it produces appears anywhere in the opening
book, including as an intermediate position of a longer named line. That makes
book-depth = the ply at which the game first leaves the book.
"""

from __future__ import annotations

import csv
import functools
from pathlib import Path

import chess

from . import config

TSV_FILES = ("a.tsv", "b.tsv", "c.tsv", "d.tsv", "e.tsv")


def _read_lines(openings_dir: Path) -> list[tuple[str, str, str]]:
    entries: list[tuple[str, str, str]] = []
    for name in TSV_FILES:
        path = openings_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"Opening book missing: {path}. Run `chess-agent vendor-openings`."
            )
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                entries.append((row["eco"], row["name"], row["pgn"]))
    # Deterministic order regardless of filesystem listing.
    entries.sort()
    return entries


def _walk(pgn: str) -> tuple[list[str], chess.Board]:
    """Replay a TSV `pgn` field (SAN with move numbers) and return each EPD along it."""
    board = chess.Board()
    epds: list[str] = []
    for token in pgn.split():
        if token.endswith("."):  # "1." / "12." move numbers
            continue
        board.push_san(token)
        epds.append(board.epd())
    return epds, board


@functools.lru_cache(maxsize=1)
def book_positions(openings_dir: str | None = None) -> dict[str, tuple[str, str]]:
    """EPD -> (eco, name) for every position reachable along a named opening line.

    Built in two passes so that a position which *is* a named line's final
    position gets that line's exact name, and only positions that are purely
    intermediate fall back to the name of a line passing through them.
    """
    directory = Path(openings_dir) if openings_dir else config.OPENINGS_DIR
    entries = _read_lines(directory)

    walked = [(eco, name, _walk(pgn)[0]) for eco, name, pgn in entries]

    positions: dict[str, tuple[str, str]] = {}
    for eco, name, epds in walked:  # pass 1: terminal (exactly named) positions
        if epds:
            positions.setdefault(epds[-1], (eco, name))
    for eco, name, epds in walked:  # pass 2: intermediate positions
        for epd in epds[:-1]:
            positions.setdefault(epd, (eco, name))
    return positions


def openings_sha(openings_dir: Path | None = None) -> str | None:
    path = (openings_dir or config.OPENINGS_DIR) / "COMMIT_SHA"
    return path.read_text().strip() if path.exists() else None


def lookup(epd: str, openings_dir: str | None = None) -> tuple[str, str] | None:
    """Return (eco, name) if this position is in the book, else None."""
    return book_positions(openings_dir).get(epd)
