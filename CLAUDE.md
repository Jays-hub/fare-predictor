# CLAUDE.md — ATL Domestic Fare-Drop Predictor

Context for any AI assistant or collaborator picking up this project.

## Goal & Scope
- Personal **buy-vs-wait recommender + mistake-fare alarm** for flights out of ATL.
- Scope: **roundtrip domestic flights from ATL, Delta and Frontier only.**
- Edge = **modeling depth on a narrow dataset**, not data breadth. ML-first.
- Budget: ~$0–5/month. Effort: ~1 week plumbing, rest on DS/ML.
- Current watch list: routes **MCO, LAS, DEN**; date-pair ladder (3-day = Fri→Mon, 7-day = Mon→Mon): **06-26→06-29**, **07-10→07-13**, **07-31→08-03** (3-day) and **07-20→07-27**, **08-17→08-24** (7-day, all 2026); economy, 1 adult.

## Three governing principles
1. **Data is time — you can't backfill it.** The collector runs first; model quality is gated by weeks collected.
2. **Judge models in dollars saved vs. "buy now," not RMSE.** Build the eval harness before any model.
3. **Scope is the edge.** Delta & Frontier compete head-to-head on ATL leisure routes, so each carrier's price predicts the other's. Protect that signal.

## Data Source (`fast-flights`) — hard-won facts
- Parses Google Flights' internal protobuf via `primp`; no official API.
- **No airline filter at code level** — `FlightData` takes only `date`/`from_airport`/`to_airport`/`max_stops`. Filter to carriers **client-side** by the `name` field.
- A **round-trip query returns only OUTBOUND flights**, each tagged with the **round-trip TOTAL price** (confirmed against the live UI).
- `Result` has `current_price` (Google's low/typical/high verdict) + `flights`.
- **Best flights are listed twice** (best section + all-flights section) → must dedup.
- `primp` chrome impersonation falls back to "random" (profiles not compiled in 1.3.1 on M-series Mac). Does **not** corrupt data — only lowers hit rate. Bot-detection blocking is **probabilistic/intermittent**.
- Degraded responses happen: prices parse but `name`/times come back empty → unusable (can't attribute carrier) → treat as soft failure and retry.
- Library's own `fetch_mode` fallback chain hits a **Turnstile 401** — do NOT use it; our retry logic replaces it. Use default `common` mode.

## Architecture & Design Decisions
- **Reliability layer** (`fetch_with_retry`): wraps `get_flights`; retries on **any raised exception** (known cases: RuntimeError "Loading results" stub, AssertionError non-200) AND soft degraded-parse; 4 attempts, exponential backoff + jitter; returns `None` on total failure (never raises) so the loop continues.
- **Second pass**: combos that fail the first pass retried once after a 30–60s cooldown to decorrelate from transient blocking.
- **GitHub runner > local Mac** for collection — x86 `primp` impersonation works there, giving higher success than the M-series laptop.
- **Storage = parallel CSV + Postgres.** CSV always written (local inspectability); Postgres written only when `DATABASE_URL` is set (env-var toggle, no flags). Neon serverless Postgres. **`ON CONFLICT DO NOTHING` + unique index** for write-time dedup.
- **Postgres is the source of truth**; the local CSV only holds rows from laptop runs (runner CSV is ephemeral). Phase 2 EDA reads from Neon. CSV→Parquet migration deferred.
- **Time handling**: `observed_at` stored UTC ISO; **one shared timestamp per collection cycle** (gives each cycle a clean snapshot identity). `days_to_dep` computed on the **UTC date** — equals the Eastern date at the 10:00/22:00 UTC snapshot times, so **no `tzdata` dependency**.
- **Store precision, derive buckets.** `dep_hour`/`dep_minute` stored exact; time-of-day buckets (redeye, pre-noon) derived at modeling time so definitions can change without recollecting. Never round/bucket in storage.
- **Date-pairs are FIXED calendar anchors**, NOT regenerated from `today()` each run — otherwise the departure date drifts daily and you never observe the same itinerary twice (breaks trajectory reconstruction).
- **Horizon ladder**: keep 3–5 staggered anchors per trip shape (consistent day-of-week shapes: Fri→Mon, Mon→Mon) so the panel always spans a spread of days-to-departure. Expired anchors are **skipped automatically** at run time (`live_date_pairs`); the freshness monitor **hard-fails when the farthest collected anchor is <14 days out** (soft-warns <21), so running out of horizon produces an email, not silent decay. Each ladder commit also resets GitHub's 60-day workflow auto-disable clock.
- **`price_level` captured per query**: Google's low/typical/high `current_price` verdict is denormalized onto every row (nullable). It can't be backfilled, and it serves as both a must-beat baseline ("buy iff Google says low") and a model feature. The collector self-applies the column migration (`ADD COLUMN IF NOT EXISTS`) so schema and code can't race; canonical DDL lives in `db/schema.sql`.
- **Two trajectory views** (`trajectories.py`): `carrier_min_series` (per-carrier cheapest fare per snapshot = headline buy-vs-wait unit, immune to flight reordering) and `departure_series` (per-departure, `flight_key` tracks one flight over time, for time-of-day modeling).
- **IO separated from transformation**: only `load_snapshots` touches the DB; all reshaping functions take/return DataFrames (testable on synthetic data, reusable by the harness).
- **Reusable module > notebook** for shared logic (harness + notebooks import it; modules diff cleanly in git).
- EDA delivered as a **percent-script (`# %%`)**, not `.ipynb` — git-friendly, runs as a notebook in VSCode.

## Conventions & Rules
- **Run all commands from the REPO ROOT** (`pip freeze`, `git`, `python scripts/collect.py`), never from inside `scripts/`. Code anchors paths via `Path(__file__)`. (Two stray artifacts — `scripts/data/` and `scripts/requirements.txt` — came from violating this.)
- **`requirements.txt`: root is the single source of truth** (the workflow installs from it); keep it **fully pinned** via `pip freeze`. No bare package names.
- **Secrets**: `DATABASE_URL` NEVER in code / repo / workflow YAML — only in a local shell `export` and **GitHub Actions secrets**. The repo is **PUBLIC**. (Connection string was leaked in chat twice and rotated — assume any string pasted into chat is burned.)
- **Public code, private data**: code is public; collected data lives only in Neon, never committed — matches the "don't redistribute the data" boundary.
- **Stay at hobby scale**: no proxies / CAPTCHA evasion. Needing them is the signal you've outgrown hobby scale, not a cue to escalate.
- **Logging discipline**: every `save_to_postgres` outcome logs one of `wrote N rows (M duplicate(s) skipped)` (real insert count via `cur.rowcount`, not attempted count) / `write failed` / `CSV only`; each cycle logs total rows + unrecoverable count + any skipped expired date-pairs.

## What's Been Built
- `scripts/validate_source.py` — Phase 0 source validation (kept as a record).
- `scripts/collect.py` — production collector: watch list (routes × date-pair ladder, expired pairs auto-skipped), `fetch_with_retry`, `parse_result` (dedup, price/duration/departure parsing, carrier filter, `price_level` stamp, schema rows), `persist` (CSV + Postgres incl. self-applied micro-migration), `run_once` (first pass + cooldown second pass).
- `scripts/trajectories.py` — reconstruction module: `load_snapshots`, `carrier_min_series`, `departure_series`, `pivot_min_by_carrier`, `itinerary_id`.
- `tests/` — pytest suite for the parsing layer and reshaping logic on synthetic data (no network/DB); `.github/workflows/test.yml` runs it on every push.
- `notebooks/eda_thesis_check.py` — thesis-validation EDA (percent-script).
- `scripts/check_freshness.py` — staleness check (hard fail at >24h) + horizon check (hard fail <14 days, warn <21) + route/carrier-coverage warnings (soft).
- `.github/workflows/collect.yml` — scheduled collector at **10:00 & 22:00 UTC** (06:00/18:00 EDT, 05:00/17:00 EST — the UTC-date `days_to_dep` logic holds in both), plus `workflow_dispatch`; `DATABASE_URL` injected from secrets.
- `.github/workflows/monitor.yml` — freshness monitor ~3h after each collection; a failed run → GitHub failure email (silence = healthy).
- **Neon Postgres `snapshots` table** with dedup unique index; DDL recorded in `db/schema.sql` (reconstructed — verify the index definition against the live DB when convenient).
- **Status**: collecting since 2026-06-04 (~2000+ rows); scheduled-run DB writes verified at both log and database level. Pipeline is unattended, self-guarding (expiry skip + horizon alarm), and healthy.

## What's Next / Open Questions
- **Run the EDA on real data** — validate the 3 theses before building on them: (A) do prices move? (B) do Delta & Frontier co-move? (C) are there upward fare steps (bucket-depletion proxy)? Needs `DATABASE_URL` exported in the shell that launches it.
- **Build the evaluation harness** — simulate a buy/wait decision at every snapshot of `carrier_min_series`; report **$ saved vs. always-buy-now** and **% of oracle (perfect-timing) savings captured**; split **by itinerary departure date** (time-aware; never split snapshots of one itinerary across train/test). The make-or-break discipline; buildable now on thin/synthetic data.
- **Mistake-fare alarm** — outlier detector vs per-route trailing baseline (e.g. 14-day min, % threshold); needs little data, fastest real-money win — ship early. Channel: reuse the proven fail→email pattern (`check_deals.py` + a `deals.yml` scheduled ~30 min after collection; exit 1 = deal found). No new notification infra.
- **Then**: naive baselines (always-buy-now; fixed days-out; buy-if-below-trailing-median; **buy-iff-Google-says-low** via `price_level`) → first model (**LightGBM** — no XGBoost bake-off, redundant at this scale; features: competing-carrier price, days-the-min-held, upward-step size/frequency, lags, rolling stats, holiday flags, `price_level`) → advanced (**survival/hazard on fare-step events** — directly models thesis C; GRU/LSTM/TFT only if the panel ever grows ~10×) → decision layer.
- **Data maturity**: meaningful EDA ~3–4 weeks; trustworthy first model ~6–8 weeks. Cannot be compressed.
- **Maintenance watch**: `fast-flights` may break when Google changes the endpoint (`pip install -U` / patch); GitHub auto-disables scheduled workflows after 60 days with no commits (ladder commits double as keep-alives); horizon refills are email-driven (monitor hard-fails <14 days) — when the email arrives, extend `DATE_PAIRS` keeping the Fri→Mon / Mon→Mon shapes; verify `db/schema.sql`'s unique-index definition against the live DB (`psql "$DATABASE_URL" -c '\d snapshots'`) and correct the file if it differs.