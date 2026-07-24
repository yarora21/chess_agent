# Decision: SQLite as the store

**Status:** accepted.

The eval cache is the project's most expensive artifact (core principle: never recompute), and it needs indexed lookups ("which games lack evals?") plus ACID transactions so a crashed parallel backfill resumes cleanly instead of leaving half-written evals. The workload is single-user, local, and embedded — SQLite's sweet spot; a client-server DB (Postgres) adds a daemon and setup friction for zero benefit here. The data is naturally relational (games → moves → evals → metrics) and the schema doubles as the inter-stage contract, which a typed schema enforces better than ad-hoc files. Scale is trivial (<1M rows even at 10k games).

**Alternatives considered:** flat files/JSON lose on indexing and corruption-safety under parallel writes. DuckDB is the closest rival for the analytics side but is weaker for incremental transactional writes during ingest/enrich, and pandas covers the analytics gap at this scale.
