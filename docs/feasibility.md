# Feasibility Assessment: A Longitudinal "How My Chess Improved" Agent from Lichess Game History

## TL;DR
- **Yes, this is technically feasible and a genuinely strong portfolio project** — every data source you need (game export, server evals, clock data, ECO codes, phase "division" markers) is available through the documented Lichess public API, and the analysis engine (Stockfish 17.1/18 via python-chess) runs fine on a laptop. Build it as a **deterministic stats pipeline with the LLM strictly as a narrator over verified numbers.**
- **The hard part is not engineering — it is statistics.** A few hundred rapid games over a few months is a genuinely thin sample for confident per-phase claims, and the **opponent-strength confound is real and severe**: as you climbed 1200→1500 your opponents got ~300 points stronger, which can flatten or worsen raw ACPL/blunder-rate even as you improved. Naive "your endgame blunder rate dropped 40%" claims will frequently be noise. This is the single biggest threat to the project's validity.
- **What makes it non-trivial (not a Lichess API wrapper):** use **Maia** (human move-prediction nets bucketed by rating) to reframe every move as "typical-1200 vs typical-1500 decision," add **opponent-adjusted metrics and proper confidence intervals / significance tests**, segment by game phase, and ship an **eval harness that verifies each generated sentence is backed by a computed statistic that clears a significance threshold.** Do those four things and it is a legitimately impressive ML systems project.

## Key Findings

### 1. Lichess data access is excellent and free
- `GET /api/games/user/{username}` streams a user's entire game history with rich filters (`since`, `until`, `max`, `rated`, `perfType`, `analysed`, `moves`, `pgnInJson`, `clocks`, `evals`, `opening`, `accuracy`, `division`, `tags`, `literate`). Output is PGN or **NDJSON** (one JSON object per line, ideal for streaming large collections).
- A **personal API access token** (created at lichess.org/account/oauth/token) is enough for your own games; export of public games needs no special OAuth scope. Tokens are long-lived (≈1 year).
- **Rate limits are deliberately unpublished** ("varied and ever changing"). The firm rules: **only make one request at a time**, and on any HTTP 429 **wait a full minute before resuming**. Authenticated requests get somewhat higher limits than anonymous. Users report hitting limits around several thousand requests over a couple of hours even when spacing them (e.g., ~4,000–5,000 requests in ~2 hours triggered 429s in one documented case).
- **Crucially, the whole-history export is a single streamed request, not one-request-per-game.** Downloading a few hundred to a few thousand games is one (or a handful of) streamed calls and takes on the order of seconds to a couple of minutes — not a rate-limit problem. The rate limit bites only if you also hit per-game or cloud-eval endpoints in a loop.

### 2. Server-side evals exist but coverage is partial
- `evals=true` returns Lichess's stored **Stockfish analysis** for games that were analyzed on the site: per-move centipawn evaluations, mate scores, and (with `literate`) textual annotations flagging inaccuracy/mistake/blunder plus best-move variations. The JSON `analysis` array carries `eval`, `best`, `variation`, and a `judgment` object (e.g. `{"name":"Blunder","comment":"d5 was best."}`).
- **This data only exists where someone requested computer analysis.** For a typical improving player, only a minority of games have it (most casual games are never analyzed). So you cannot rely on server evals for full coverage — **you will need to generate your own evals** for the rest.
- There is **no public API to trigger** "request computer analysis" on your games; it is a web-UI action with per-user daily/weekly caps (one user reported a 30/day, 120/week cap). So programmatically you either use the evals that happen to exist, query the **cloud-eval** endpoint position-by-position, or run Stockfish yourself.

### 3. You can and should generate your own evals with Stockfish
- **Stockfish 17.1 released March 30, 2025** (per the official Stockfish blog: "an Elo gain of up to 20 points and winning close to 2 times more game pairs than it loses"), and **Stockfish 18 released January 31, 2026**. All modern versions are NNUE-only and GPL-3.0. Driven from Python via `python-chess`'s `chess.engine.SimpleEngine.analyse(board, Limit(...))`.
- **Throughput (from targeted research):** single-thread NNUE runs roughly 1–2.5 Mnps on a modern laptop core (e.g., ~1.29 Mnps single-thread on a Ryzen 9 3950X per Stockfish's own "Useful data" wiki; ~2.3 Mnps single-thread on an Apple M1). python-chess pipe overhead is negligible vs. search time. Practical wall-clock per ~80-half-move game: **~8–10 s at 0.1 s/move; ~80 s at 1 s/move; ~10–40 s at fixed depth 15; ~1.5–7 min at depth 20** (position-dependent). Sequentially, **1000 games at 0.1 s/move ≈ 2 hours; at 1 s/move ≈ ~22 hours.**
- **Parallelize with one single-threaded engine per physical core** (`ProcessPoolExecutor`), NOT many threads on one position — Stockfish's Lazy SMP scales node count near-linearly but time-to-depth poorly (measured cases show 4× cores reaching roughly the *same* depth). On an 8-core laptop this gives ~8× throughput, so 1000 games at 1 s/move drops to ~3 hours.
- **At the 1200–1500 band, errors are large, so shallow analysis is sufficient.** Real tools default low: python-chess-annotator uses depth 14, serg-meus/pgn_err_stats defaults to 0.5 s/move, chess-artist to ~2 s/move. Depth 12–15 or ~0.1–0.3 s/move reliably catches the ~200+ centipawn blunders that decide games at this level. Save depth 18–20 for a refinement pass if needed.

### 4. Lichess's own metric definitions are documented and reproducible
- **Win% from centipawns:** `Win% = 50 + 50*(2/(1+exp(-0.00368208*centipawns)) - 1)`.
- **Move Accuracy%:** `Accuracy% = 103.1668*exp(-0.04354*(winPercentBefore - winPercentAfter)) - 3.1669`.
- **Game accuracy** is not a simple mean: Lichess uses sliding windows, weights move accuracies by the volatility (standard deviation of Win%) within each window, then takes a harmonic mean.
- **ACPL** (average centipawn loss) is the mean per-move drop in engine eval, conventionally capped so that losses in already-won/lost positions don't dominate.
- Inaccuracy/mistake/blunder classification is **win%-based, not raw-centipawn-based** — the same "winning chances" metric that drives the eval bars. This matters: it correctly treats "+9 to +6" as no blunder while flagging "+0.5 to −2.5." You should replicate this win%-delta thresholding rather than fixed centipawn cutoffs.
- **Known limitation you must state honestly:** accuracy/ACPL depend on how the opponent played (blunders in already-winning positions inflate your accuracy), differ across engine depth and platform, and are averages of regret — a single hung-piece game and forty tiny imprecisions can produce identical ACPL with opposite diagnoses.

### 5. Maia is the key differentiator over raw Stockfish
- **Maia** (CSSLab, KDD 2020) is a set of **nine human move-prediction neural nets, Maia-1100 through Maia-1900 in 100-point steps, each trained on ~12 million Lichess games**, matching human moves over 50% of the time (~46–52% top-1 move-matching). It runs on the Lc0 backend in nodes=1 (no-search) UCI mode. **Maia-2** (Tang et al., NeurIPS 2024, arXiv:2409.20553) unifies skill levels into one model using a "skill-aware attention mechanism to dynamically integrate players' strengths with encoded chess positions," spanning ~600–2600; the official CSSLab implementation is Python-installable and supports Python 3.10–3.12 on CUDA/MPS/CPU (github.com/CSSLab/maia2). **Maia-3 / Chessformer** (ICLR 2026) is the current most-accurate human predictor.
- **Why this beats Stockfish for your use case:** Stockfish tells you the *best* move; Maia tells you **what a human at rating X would most likely play**. That directly enables your core claim type — *"this was a typical 1200 mistake that a typical 1500 avoids"* — by comparing your move's probability under Maia-1200 vs Maia-1500. It converts an abstract engine loss into a human-relative skill statement, which is exactly what "how did my play improve" needs.
- Maia is GPL-3.0 (compatible with a personal/portfolio project; note copyleft if you ever distribute).

### 6. Existing tools mostly do current-weakness snapshots, not longitudinal narration
- **Aimchess** is the closest competitor: acquired by Play Magnus Group in 2021 and then by Chess.com when it bought Play Magnus Group in a ~$82.5M deal that closed December 16, 2022; it reports 100,000+ registered users since inception. It ingests your Lichess/Chess.com history, produces strength/weakness reports across dimensions (e.g. "Resourcefulness"), compares you to peers in your rating range, and turns your mistakes into targeted puzzles. Free tier = one 40-game report/month; Premium ≈ $7.99/mo monthly or ≈ $4.85/mo annual, adding deeper analysis of up to ~1000 recent games. Reviews call it a solid **diagnostic** ("the doctor ordering the right tests"), but users debate whether it justifies the cost, and it is primarily a **current-snapshot + training** tool.
- **Lichess Insights** is a free, powerful "answer engine" over your own games (metric × dimension × filters: move times, phase detection, material imbalances, ACPL, "opportunity"/"luck"). It is **web-only with no dedicated public API** — but you can reproduce its logic yourself from the raw export, which is part of what makes your project non-trivial.
- **Chess.com Game Review / Insights, DecodeChess** (explains *why* a move is good in plain English), and various LLM-coaching products exist. DecodeChess is single-position explanation, not longitudinal.
- **The specific gap your project targets — automated LONGITUDINAL improvement narration ("compared to 3 months ago you now blunder 40% less in the endgame") — is not something these tools do well or foreground.** That is your differentiated angle, provided you solve the statistics.

### 7. LLMs cannot be trusted to understand chess — use them only as narrators
- Independent 2025–2026 evidence is consistent: LLMs hallucinate illegal moves and misread positions once out of opening book. The **Kaggle Game Arena chess exhibition (Aug 5–7, 2025, run with Google DeepMind) allowed "up to three retries to play a legal move at each turn (for a total of four attempts)"** and models that still failed lost the game; Mathieu Acher documented forcing **GPT-5 and GPT-5 Thinking into an illegal move after the 4th turn** ("Illegal Move After 4th Turn"). Frontier models play at weak-club-player level and degrade sharply out of book — Magnus Carlsen defeated ChatGPT in July 2025 without losing a single piece.
- **Implication:** the LLM must never *judge* chess. All chess truth (what was a blunder, what the better move was, which phase, how much a metric changed) comes from the deterministic Stockfish/Maia/stats layer. The LLM only **verbalizes pre-computed, verified statistics** — the classic "data-to-text" problem, where the well-documented failure mode is hallucinating facts unsupported by the source table. Mitigations from the literature: constrain generation to provided structured facts, template/slot scaffolding, and post-hoc faithfulness verification.

## Details

### Recommended Python stack
- **Data:** `berserk` (the official Lichess-org-maintained Python client; actively maintained, requires Python 3.10+, wraps `games.export`, `games.export_by_player`, `users.get_rating_history`, `analysis.get_cloud_evaluation`; handles NDJSON/PGN and token auth). Fall back to raw `requests` for any endpoint berserk lags on.
- **Parsing/board logic:** `python-chess` (PGN parsing, FEN, legal-move generation, phase heuristics, and the UCI engine interface). Mature and canonical.
- **Engine:** Stockfish 17.1/18 (local binary) for your own evals; optionally the Lichess `cloud-eval` endpoint for opening/common positions (single FEN per query, rate-limited — not a batch tool).
- **Human model:** Maia via Lc0 (rating-bucket nets) or Maia-2 (`pip` install, PyTorch).
- **Stats:** `pandas`, `numpy`, `scipy`/`statsmodels` for confidence intervals, bootstrap, and significance testing.
- **Narration:** any capable LLM behind a strict grounded-generation harness (your LangGraph/FastAPI comfort maps directly here).
- **Rating history:** `GET /api/user/{username}/rating-history` returns a per-perf time series of `[year, month(0-indexed), day, rating]` points — a daily-granularity ladder per time control, perfect for anchoring "over this window your rapid went 1200→1500" and for computing opponent-strength context.

### The extractable metrics that can support plain-English claims
Given per-move evals + clocks + ECO + `division` (phase boundaries), you can compute, per time window:
- Blunder/mistake/inaccuracy rate **by phase** (opening/middlegame/endgame) using win%-delta thresholds.
- ACPL by phase (capped).
- **Time management:** using clock comments — do blunders cluster when time remaining is low? (a genuinely robust, low-noise signal at your level).
- Opening repertoire consistency and results by ECO code; first-out-of-book move quality.
- **Conversion rate** of winning positions (games where eval reached ≥ +2 for you — did you win?) and **defensive resilience** in losing positions.
- **Recovery:** after an opponent blunder, how often you find the refuting/best continuation.
- Maia-relative metrics: fraction of your moves matching Maia-1200 vs Maia-1500 predictions; "skill percentile" of your move choices by phase.

### The statistical core — be skeptical here
- **Sample size:** community practice suggests ≥30–40 games just to make aggregate ACPL "statistically relevant," and that is for a *single* number. Once you slice by **phase × time-window**, each cell may hold only a few dozen relevant move-decisions. Rare events (endgame blunders) have high variance; a "40% reduction" across two ~150-game windows can easily be within noise.
- **What actually helps:** (a) aggregate at the **move-decision** level, not the game level — Regan's Intrinsic Performance Rating work notes 50 games ≈ ~1,500 move decisions, a far healthier sample; (b) attach **bootstrap confidence intervals** to every metric and only narrate differences whose CIs separate; (c) prefer **high-frequency, low-variance signals** (time-trouble blunders, opening-move accuracy, move-matching rates) over rare-event rates.
- **The opponent-strength confound is the headline risk.** Rising from 1200→1500 means your opponents strengthened by ~300 Elo, which independently pushes your raw ACPL/blunder rate *up*. So flat or worsening raw metrics can hide real improvement, and naive comparisons are systematically biased. **Fixes:** (1) Maia move-matching is largely opponent-independent (it scores *your* decision quality against a fixed human baseline, not the game result); (2) Regan-style intrinsic performance rating estimates strength from move quality benchmarked against a strong engine, explicitly designed to be outcome- and opponent-independent; (3) stratify/regress metrics on opponent rating and report opponent-adjusted trends. **Say this plainly in the product: raw blunder-rate deltas are confounded; the trustworthy signals are the opponent-adjusted and Maia-relative ones.**

### Architecture: deterministic pipeline, LLM as narrator
Recommended (most reliable AND most impressive):
1. **Ingest** — stream full history via berserk → store PGN + JSON (SQLite/parquet).
2. **Enrich** — for each game: parse with python-chess; attach existing server evals; fill gaps with local Stockfish (parallel, depth ~14 / 0.1–0.3 s); run Maia-1200…1500 to get human move probabilities.
3. **Metrics** — compute per-move features; aggregate by phase × time-window; compute opponent-adjusted and Maia-relative metrics; attach bootstrap CIs and run significance tests (e.g., two-proportion tests for rates, Mann–Whitney for distributions).
4. **Claim generation (deterministic)** — emit a structured list of *candidate claims* only where a metric change clears a pre-set significance/effect-size threshold. Each claim is a typed record: {metric, phase, window_a, window_b, delta, CI, p, supporting_game_ids}.
5. **Narration (LLM)** — the LLM receives ONLY the verified claim records and writes plain-English sentences, forbidden from introducing any chess judgment not in the records. Optionally an agentic layer lets the LLM *query* the stats store (tool calls) but still only over verified aggregates.
- **Deterministic-pipeline-with-LLM-narrator beats free agentic exploration** on reliability. For a portfolio, implement the deterministic core first, then add a constrained agentic query layer as a showcase — but keep the guardrail that the LLM never computes or judges chess.

### The eval harness (this is your differentiator for an ML-focused portfolio)
- **Claim-level faithfulness eval:** every generated sentence must map to exactly one verified claim record; automatically parse the narrative and assert each numeric/factual assertion matches a record (reject hallucinated numbers). This is a grounded-generation/faithfulness check straight from the data-to-text literature.
- **Statistical-validity eval:** unit-test that no claim is emitted whose CI includes zero or whose p exceeds threshold; include a **placebo/permutation test** — shuffle game timestamps and confirm the system produces (near) zero "significant improvements," proving it isn't manufacturing narratives from noise.
- **Backtest:** hold out the most recent window, "predict" improvement, and check it against actual later rating movement.
- **Human/reference check:** spot-check a sample of flagged blunders against Lichess's own server analysis where it exists.

## Recommendations

**Verdict: Build it — but scope it around the statistics, not the API.** The data plumbing is a solved, one-weekend problem; the intellectual contribution (and the thing that separates this from a thin API wrapper) is opponent-adjusted, Maia-relative, significance-gated longitudinal claims with a faithfulness eval harness.

Staged build plan (rough estimates for an experienced engineer):
1. **Week 1 — Ingestion + storage.** berserk pull of full history to SQLite/parquet; parse PGN with python-chess; extract clocks, ECO, `division`, existing evals, rating-history. *Benchmark to move on:* can reconstruct every game's board states and phase boundaries.
2. **Week 1–2 — Eval generation.** Parallel Stockfish (depth ~14, one engine/core) to fill missing evals; reproduce Lichess Win%/Accuracy% and win%-delta blunder classification. *Benchmark:* your blunder counts match Lichess server analysis within tolerance on games that have both.
3. **Week 2–3 — Metrics + statistics.** Phase × window aggregation; bootstrap CIs; significance tests; opponent-rating stratification. *Benchmark:* placebo/permutation test yields ~no false "improvements."
4. **Week 3–4 — Maia integration.** Move-matching vs Maia-1200/1500; opponent-independent skill metrics. *Benchmark:* Maia-relative trend correlates with your actual rating gain.
5. **Week 4–5 — Narration + eval harness.** Deterministic claim records → constrained LLM narration; faithfulness parser; statistical-validity gates. *Benchmark:* zero hallucinated numbers over a test set; every sentence traces to a record.
6. **Optional Week 6 — Agentic query layer + FastAPI/LangGraph demo UI.**

**Thresholds that change the plan:**
- If, after Step 3, **fewer than ~150–200 relevant move-decisions per phase-window cell**, widen windows (e.g., halves rather than months) or drop phase granularity — don't narrate underpowered cells.
- If Maia move-matching and opponent-adjusted ACPL **disagree in direction**, trust the opponent-adjusted/Maia signals over raw ACPL and say so.
- If your total analyzed sample is under ~100 games, present **descriptive** findings with explicit uncertainty, not causal "you improved" claims.

## Caveats
- **Rate limits are undocumented and dynamic;** budget generous backoff. The bulk export is cheap, but any per-position cloud-eval loop will hit limits fast — prefer local Stockfish for batch work.
- **Server-eval coverage is partial and self-selected** (analyzed games skew toward losses/interesting games), which can bias any metric computed only over analyzed games — another reason to generate your own evals uniformly.
- **Stockfish wall-clock numbers in this report are largely derived estimates** (measured NPS × moves), not a single published "games/hour" benchmark; validate on your own hardware before committing to depth/time settings.
- **The improvement signal may genuinely be too weak** at a few-hundred-game scale for confident per-phase causal claims. If so, the honest product is: report the one or two signals that *are* significant (often time-management and opening consistency), quantify uncertainty everywhere else, and lean on Maia-relative skill percentile as the most robust longitudinal indicator. Do not let the LLM paper over thin data with fluent prose — that is the failure mode this whole architecture exists to prevent.
- **Maia rating buckets are coarse** (100-point bands) and reflect Lichess populations, not you specifically; per-player fine-tuning exists in the research (McIlroy-Young 2022) but is beyond a first build.
- Some third-party client libraries (python-lichess, lichesspy, older berserk forks) vary in maintenance; prefer the lichess-org/berserk repo and verify endpoint coverage against the current OpenAPI spec, which changes frequently.