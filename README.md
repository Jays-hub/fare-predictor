# ATL Fare-Drop Predictor

Personal buy-vs-wait recommender and mistake-fare alarm for roundtrip domestic
flights out of ATL (Delta & Frontier only — they compete head-to-head on ATL
leisure routes, so each carrier's price helps predict the other's). Fares are
snapshotted twice daily from Google Flights via
[fast-flights](https://github.com/AWeirdDev/flights) and accumulated into
price trajectories; models are judged in **dollars saved vs. buying
immediately**, not RMSE.

## Layout

- `scripts/collect.py` — scheduled collector (GitHub Actions, 10:00/22:00 UTC)
- `scripts/trajectories.py` — snapshot → trajectory reshaping (DB-free, testable)
- `scripts/check_freshness.py` — pipeline health monitor (failure → email)
- `notebooks/eda_thesis_check.py` — thesis-validation EDA (percent-script)
- `db/schema.sql` — Postgres schema incl. the dedup index
- `tests/` — parsing + reshaping tests on synthetic data (`pytest`)
- `CLAUDE.md` — full project context, design decisions, and roadmap

## Boundaries

**Code is public; collected data is not.** Fares live in a private Postgres,
are never committed, and are not redistributed. Collection is deliberately
hobby-scale: a handful of paced queries per day, with retries but no proxies
or bot-detection evasion.
