# Chess Improvement Agent

Analyzes a Lichess **rapid** history and reports where the play changed as the rating moved — per phase, per skill dimension — then answers follow-up questions about it.

The point of the project is not the chess analysis. It is the discipline around a language model: **the LLM never judges chess.** Every blunder, phase boundary, and rate comes from a deterministic Stockfish/Maia/stats layer. The model is a narrator over a table of verified numbers, and a harness rejects its output if it invents one.

Run against the author's history: **724 games, 46,143 engine-evaluated positions, 46,410 Maia predictions.**

---

## What it found

First 100 games vs last 100, over a run from ~1140 to ~1450:

| metric | first 100 | last 100 | sample |
|---|---|---|---|
| blunder rate (overall) | 8.9% | 7.4% | 3,089 / 3,143 moves |
| blunder rate (opening) | 5.1% | **6.1%** ↑ | 1,071 / 1,111 |
| blunder rate (middlegame) | 13.1% | 9.3% | 1,318 / 1,349 |
| blunder rate (endgame) | 6.6% | 5.7% | 700 / 683 |
| conversion of winning positions | 72.3% | 80.0% | 65 / 50 games |
| average centipawn loss | 66.6 | 62.9 | 100 / 100 games |
| book depth | 3.5 plies | 4.9 plies | 100 / 100 games |

**These are descriptions, not verdicts.** The MVP ships without significance testing (see [Known gaps](#known-gaps)), so some of these differences are noise. The middlegame movement rests on the largest sample; the time-trouble figures rest on ~100 moves and should not be read as a finding.

### The Maia result is negative, and that's reported as such

Maia models predict what a human of a given rating would play. The obvious metric — "how often do I play the Maia-1500 move?" — turned out to be **nearly useless**: the 1200 and 1500 nets predict the **same move on 74.6%** of positions. Those are obvious moves every rating band plays, and they dominate the match rate, which is why both rates track each other (43.7→46.2% and 44.0→46.9%) and neither separates skill levels.

Restricting to the 5,891 positions where the nets *disagree* isolates the moves that actually discriminate. There, the share of "1500 moves" went 51.3% → 52.5% across the run — a shift small enough, on ~450 decisive moves per window, to read as **no detectable change**.

Both the failed metric and its replacement are in the codebase. A negative result reported honestly was preferred over the flattering raw number.

---

## Design principles

1. **The LLM never judges chess.** All chess truth comes from the deterministic layer. The model only verbalizes pre-computed records.
2. **Sample size travels with every number.** Without significance testing, the denominator is the only thing stopping a 751-move finding from reading like a 23,080-move one.
3. **Analyze each game once per (game, engine-config).** Evals are cached by game *and* engine identity. A metrics run reads from exactly one config — mixing them would let eval quality correlate with analysis order and manufacture a trend.
4. **Beware the opponent-strength confound.** Rising 1140→1450 means opponents got ~300 Elo stronger, which pushes raw blunder rates *up* even as play improves. Raw deltas are supporting evidence only.

---

## Pipeline

```
ingest → enrich → metrics → narrate
                         ↘ ask (agentic Q&A over the store)
```

| Stage | What it does |
|---|---|
| **ingest** | One streamed Lichess API call → SQLite. Rapid only; mixing time controls is a correctness bug, not a setting. |
| **enrich** | Stockfish 18 at fixed depth 14, one single-threaded engine per core. Maia-1200/1500 via Lc0 at `nodes=1` (policy head, no search). |
| **metrics** | Per-game blunder rates by phase, capped ACPL, time-trouble splits, book depth, conversion. |
| **narrate** | Records → Claude → prose, with a faithfulness harness. |
| **ask** | Read-only agentic Q&A citing game IDs. |

### Three decisions worth explaining

**Fixed depth, not fixed movetime.** A correctness choice. Fixed movetime makes eval quality a function of machine load, and games are handed to the worker pool in chronological order — so a busy patch degrades a *contiguous run of games*, indistinguishable from a real trend in play quality. Fixed depth is reproducible regardless of what else the laptop is doing.

**Local Stockfish on every move, even where Lichess had already analysed.** 40% of the games carried free server evals. Tempting — but coverage is strongly non-random in time (22%, 9%, 10%, 18%, 58%, 66%, 83%, 73%, 39%, 20% by decile; r=+0.24, p=1.3e-11). Mixing eval sources would make provenance a function of time, so any systematic difference between them would appear as improvement. All 46,143 positions were evaluated locally under one config; Lichess's 292 analysed games became the validation set instead.

**Blunder thresholds fitted, not invented.** The 307 Lichess-analysed games contain their own verdicts on 21,157 moves. Fitting a win%-loss rule against those labels puts the blunder cut at **15 win% lost**, agreeing on 1,661 of 1,664 flagged moves (3 false positives).

---

## Verification

The project is ~40% test and gate code that produces no user-facing output. That's deliberate: the premise is telling you true things about your chess, so a quiet data bug wouldn't raise an error — it would produce a confident, wrong report.

**Phase 1 gate** (`chess-agent verify`) replays all 770 games and checks stored positions, results, phase monotonicity, and index density. It caught a real bug: the gate initially assumed book moves form a prefix. They don't — 57 games transpose back into known theory after leaving it.

**Phase 2 gate** (`chess-agent validate`) compares blunder counts against Lichess's own on 292 games: **1,550 vs 1,697, −8.7%**, inside the ±10% tolerance agreed *before* implementation. Per-game counts correlate at r=0.951. Critically, the shortfall does not drift with time (r=+0.015, p=0.80) — it's flat noise, not a manufactured trend.

**Faithfulness harness** parses the generated report and asserts every number maps to a record. It caught two of its own defects: whitelisting `int(v)` let a model write "4.0" against a record of 4.1, and a regex read the hyphen in "Maia-1200" as a minus sign — failing the first real run on an otherwise-correct report.

**The Q&A agent's connection is `mode=ro`.** An instruction not to write is not a control; an LLM with a writable handle is one prompt injection away from mutating the evidence it cites. Tests assert `DELETE` and `CREATE TABLE` both raise, and that a `DROP TABLE` in a column name is refused.

---

## Usage

```bash
pip install -e .
export LICHESS_TOKEN=...        # lichess.org/account/oauth/token, no scopes needed
export ANTHROPIC_API_KEY=...    # narrate / ask only

chess-agent ingest              # one streamed call; ~14s for 770 games
chess-agent verify              # phase-1 reconstruction gate
chess-agent benchmark           # time engine settings on this machine
chess-agent backfill            # Stockfish over every move (~11 min on 8 cores)
chess-agent validate            # agreement gate vs Lichess
chess-agent metrics             # per-game metrics + summary
chess-agent maia                # Maia move-matching (~2 min)
chess-agent narrate             # LLM report + faithfulness check
chess-agent ask "which games did I blunder most in?"
```

Requires Python 3.10, a Stockfish binary, and Lc0 with Maia weights for the Maia stage.

---

## Known gaps

Deliberate, not oversights:

- **No significance testing.** Cut from the MVP to prioritize the LLM and agentic layers. With ~20 metric × phase cells and no filter, expect some narrated findings to be noise. The narrator is held to descriptive phrasing and forbidden from words like "significantly" to keep the output honest about this. Restoring it — game-level bootstrap plus a shuffled-label placebo test — is the first post-MVP task.
- **Cohort start point selects on the outcome.** Analysis begins at game 44, the rating trough. Starting a story at the minimum flatters any subsequent climb. Trends regress on game index rather than rating, which limits the damage, but narration quoting a rating span inherits the bias.
- **Maia trained on blitz, applied to rapid.** Accepted approximation at this rating band.
- **No backtest.** Holding out recent games and checking the predicted direction against actual subsequent rating movement is the strongest available validity check, and it is not implemented.

## Further reading

- [`docs/feasibility.md`](docs/feasibility.md) — the research this was built from: API capabilities, engine throughput, why Maia is the differentiator
- [`docs/decisions/sqlite.md`](docs/decisions/sqlite.md) — why SQLite
- [`chess_agent/schema.sql`](chess_agent/schema.sql) — the contract between pipeline stages, with migration notes
