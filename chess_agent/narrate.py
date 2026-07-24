"""Turn verified metric records into plain English, with a faithfulness harness.

The LLM is a narrator over a table of numbers, nothing more. It never sees a
chess position, never judges a move, and never computes anything. It receives
pre-computed records and writes sentences about them.

The harness is the enforcement: every number in the generated text must map to a
value in the records. A number that does not is a hallucination, and the run
fails rather than shipping it. That check is what makes an LLM safe to put in
front of this data at all -- especially now that significance testing is cut,
since nothing else downstream would catch an invented figure.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import maia, metrics, store

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """\
You are a narrator over a table of pre-computed chess statistics. You are not a \
chess engine, a coach, or an analyst.

ABSOLUTE RULES:
1. Every number you write MUST appear in the records provided. Never compute, \
round differently, estimate, or infer a number. If you want to say a rate \
"halved", check the records support it; if not, don't say it.
2. Never make a chess judgment. Do not name openings, suggest moves, explain \
why a rate changed, or speculate about the player's strengths. You do not have \
that information and must not invent it.
3. These records are DESCRIPTIVE ONLY. They carry no significance testing. \
Never write "significantly", "proves", "clearly shows", or "you improved". \
Write what the numbers did: "your middlegame blunder rate went from X to Y".
4. Always give the sample size when quoting a rate, and say plainly when a \
sample is small. A finding on a few hundred moves deserves a hedge; one on \
twenty thousand does not.
5. If two records point in opposite directions, say so. Do not resolve the \
tension by picking the flattering one.

Write 4-8 short paragraphs. Plain language, no jargon, no headers. Address the \
player as "you". Do not open with a preamble like "Here is your report" -- start \
with the substance.
"""


@dataclass
class Record:
    """One verified fact the narrator is allowed to talk about."""

    id: str
    text: str
    values: list[float] = field(default_factory=list)

    def render(self) -> str:
        return f"[{self.id}] {self.text}"


def build_records(conn: sqlite3.Connection, window: int = 100) -> list[Record]:
    """Assemble every fact the narrator may use. Nothing else reaches the model."""
    records: list[Record] = []
    info = store.cohort_summary(conn)
    games = conn.execute(
        f"SELECT player_rating FROM games WHERE {store.COHORT_WHERE} ORDER BY game_index",
        store.cohort_bounds(),
    ).fetchall()
    ratings = [g["player_rating"] for g in games if g["player_rating"]]

    records.append(Record(
        "cohort",
        f"The analysis covers {info['cohort_games']} rapid games containing "
        f"{info['cohort_moves']} half-moves. {info['excluded_games']} further games "
        f"were excluded as unrepresentative. Rating over this period ranged from "
        f"{info['rating_min']} to {info['rating_max']}, starting at {ratings[0]} "
        f"and ending at {ratings[-1]}.",
        [info["cohort_games"], info["cohort_moves"], info["excluded_games"],
         info["rating_min"], info["rating_max"], ratings[0], ratings[-1]],
    ))
    records.append(Record(
        "window",
        f"Comparisons below are the first {window} games against the last {window} games.",
        [window],
    ))

    for i, d in enumerate(metrics.describe(conn, window=window)):
        if d.first_value is None or d.last_value is None:
            continue
        phase = f" in the {d.phase}" if d.phase else ""
        unit = "%" if d.unit == "%" else f" {d.unit}"
        records.append(Record(
            f"m{i}",
            f"{d.metric}{phase}: first {window} games {d.first_value:.1f}{unit}, "
            f"last {window} games {d.last_value:.1f}{unit}, all games "
            f"{d.overall:.1f}{unit}. Sample: {d.first_n} and {d.last_n} "
            f"observations respectively, {d.total_n} overall.",
            [round(d.first_value, 1), round(d.last_value, 1), round(d.overall, 1),
             d.first_n, d.last_n, d.total_n, window],
        ))

    for r in maia.match_rates(conn, window=window):
        if r["first"] is None:
            continue
        records.append(Record(
            f"maia{r['rating']}",
            f"Maia-{r['rating']} move-match rate (how often you played the move a "
            f"human at that rating plays): first {window} games {r['first']:.1f}%, "
            f"last {window} games {r['last']:.1f}%, all games {r['overall']:.1f}%, "
            f"over {r['total_n']} moves.",
            [round(r["first"], 1), round(r["last"], 1), round(r["overall"], 1),
             r["rating"], r["total_n"], window],
        ))

    d = maia.discriminating_rates(conn, window=window)
    if d:
        f, l, o = d["first"], d["last"], d["overall"]
        records.append(Record(
            "maia_discriminating",
            f"On the {o['disagreements']} positions where Maia-{d['low']} and "
            f"Maia-{d['high']} would play DIFFERENT moves -- the only positions that "
            f"distinguish the two levels -- you chose the {d['high']} move "
            f"{f['high_share']:.1f}% of the time in the first {window} games and "
            f"{l['high_share']:.1f}% in the last {window}, {o['high_share']:.1f}% "
            f"overall. The two models agree on the other positions, which are "
            f"ordinary moves any level plays.",
            [o["disagreements"], d["low"], d["high"], round(f["high_share"], 1),
             round(l["high_share"], 1), round(o["high_share"], 1), window],
        ))

    return records


# ---------------------------------------------------------------------------
# Faithfulness harness
# ---------------------------------------------------------------------------

# Digits only. The sign is decided separately, because a hyphen inside a word
# ("Maia-1200", "opus-5") is punctuation, not a minus -- treating it as one made
# the harness report -1200 as an invented number in an otherwise-correct report.
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _is_negative(text: str, start: int) -> bool:
    """A leading '-' counts as a minus only when it isn't joining two words."""
    if start == 0 or text[start - 1] != "-":
        return False
    return start - 2 < 0 or not text[start - 2].isalnum()

# Numbers that are ordinary English rather than claims about the data. Kept
# deliberately tiny: every addition here is a hole in the check.
ALLOWED_BARE = {0, 1, 2, 3}


@dataclass
class Faithfulness:
    numbers_checked: int = 0
    unsupported: list[str] = field(default_factory=list)
    banned_phrases: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.banned_phrases

    def render(self) -> str:
        lines = [f"Checked {self.numbers_checked} numbers in the narration."]
        if self.unsupported:
            lines.append(f"{len(self.unsupported)} NOT found in any record:")
            lines += [f"  - {u}" for u in self.unsupported]
        if self.banned_phrases:
            lines.append(f"{len(self.banned_phrases)} forbidden phrase(s):")
            lines += [f"  - {p}" for p in self.banned_phrases]
        if self.ok:
            lines.append("PASS: every number traces to a record.")
        else:
            lines.append("FAIL: narration contains unsupported content.")
        return "\n".join(lines)


# Claims of significance the records cannot support, now that testing is cut.
BANNED = ("significant", "significantly", "statistically", "proves", "proven",
          "definitely", "conclusively", "p-value", "confidence interval")


def check(text: str, records: Iterable[Record]) -> Faithfulness:
    """Assert every number in `text` appears in `records`, and no verdict language."""
    allowed: set[float] = set()
    for record in records:
        for value in record.values:
            value = float(value)
            allowed.add(round(value, 1))
            # Only whole numbers get an integer alias (23080.0 -> 23080). Adding
            # int(4.1) would whitelist "4.0", letting the model shave a decimal
            # off any figure and still pass -- exactly the quiet drift this
            # harness exists to catch.
            if value == int(value):
                allowed.add(float(int(value)))

    report = Faithfulness()
    for match in NUMBER_RE.finditer(text):
        raw = match.group().replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        if _is_negative(text, match.start()):
            value = -value
            raw = "-" + raw
        report.numbers_checked += 1
        if value in ALLOWED_BARE and value == int(value):
            continue
        if round(value, 1) in allowed or float(int(value)) in allowed:
            continue
        context = text[max(0, match.start() - 40):match.end() + 20].replace("\n", " ")
        report.unsupported.append(f"{raw}  (...{context.strip()}...)")

    lowered = text.lower()
    for phrase in BANNED:
        if phrase in lowered:
            report.banned_phrases.append(phrase)
    return report


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def narrate(records: list[Record], model: str = MODEL) -> str:
    """Ask Claude to verbalize the records. Streamed -- the report can run long."""
    import anthropic

    client = anthropic.Anthropic()
    body = "\n".join(r.render() for r in records)
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Here are the verified records. Write the report.\n\n{body}",
        }],
    ) as stream:
        message = stream.get_final_message()

    if message.stop_reason == "refusal":
        raise RuntimeError("model declined to generate the report")
    return "".join(b.text for b in message.content if b.type == "text")
