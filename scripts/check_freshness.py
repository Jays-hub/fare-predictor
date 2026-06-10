import logging
import os
import sys
from datetime import datetime, timezone

try:
    import psycopg
except ImportError:
    psycopg = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("freshness")

# Collection runs ~2x/day (10:00 and 22:00 UTC), so healthy data is never more
# than a few hours old at check time. >24h means roughly two consecutive cycles
# produced nothing — a real failure, not a single tolerable transient miss.
MAX_AGE_HOURS = 24

# We collect 3 routes; a healthy latest cycle should cover all of them.
EXPECTED_ROUTES = 3

# Both carriers should appear in every cycle. The collector filters on exact
# name strings, so if Google ever changes a display name (say "Frontier" ->
# "Frontier Airlines") that carrier vanishes silently while rows keep flowing
# from the other — this warning is the tripwire for the co-movement signal.
EXPECTED_CARRIERS = {"Delta", "Frontier"}

# Horizon guard: DATE_PAIRS are fixed calendar anchors, so they expire as
# departures pass. Once the farthest anchor being collected is closer than
# these thresholds, the days-to-departure spread is about to collapse and new
# pairs must be added to scripts/collect.py. Soft = warn in logs; hard = fail
# the workflow, which is what actually produces an email.
HORIZON_SOFT_DAYS = 21
HORIZON_HARD_DAYS = 14


def check() -> int:
    """Return 0 if the collector looks healthy, 1 if data is stale/missing.
    The non-zero exit fails the monitor workflow and triggers GitHub's failure
    email — silence means healthy, an email means look at the collector."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        log.error("DATABASE_URL not set; cannot check freshness")
        return 1
    if psycopg is None:
        log.error("psycopg not installed; cannot check freshness")
        return 1

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT max(observed_at) FROM snapshots")
            (newest,) = cur.fetchone()
            if newest is None:
                log.error("no rows in snapshots at all — collector has never written")
                return 1
            cur.execute(
                "SELECT count(DISTINCT dest) FROM snapshots WHERE observed_at = %s",
                (newest,),
            )
            (routes,) = cur.fetchone()
            cur.execute(
                "SELECT DISTINCT carrier FROM snapshots WHERE observed_at = %s",
                (newest,),
            )
            carriers = {c for (c,) in cur.fetchall()}
            # ::date keeps this working whether dep_date is a date or text column.
            cur.execute(
                "SELECT max(dep_date::date) - CURRENT_DATE FROM snapshots WHERE observed_at = %s",
                (newest,),
            )
            (horizon_days,) = cur.fetchone()

    age_hours = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
    log.info("newest snapshot: %s (%.1f h ago), routes in that cycle: %d/%d, horizon: %s day(s)",
             newest.isoformat(timespec="seconds"), age_hours, routes, EXPECTED_ROUTES, horizon_days)

    # Coverage is a soft signal: one route failing a single cycle is within
    # tolerance, so warn but don't fail on it.
    if routes < EXPECTED_ROUTES:
        log.warning("latest cycle covered only %d of %d routes", routes, EXPECTED_ROUTES)

    # Same for carriers — but a PERSISTENT absence means the name filter broke.
    missing_carriers = EXPECTED_CARRIERS - carriers
    if missing_carriers:
        log.warning("latest cycle has no rows for: %s — if this persists, check "
                    "the carrier name filter in collect.py", ", ".join(sorted(missing_carriers)))

    if horizon_days is not None and HORIZON_HARD_DAYS <= horizon_days < HORIZON_SOFT_DAYS:
        log.warning("horizon shrinking: farthest anchor departs in %d days — "
                    "add future DATE_PAIRS to scripts/collect.py soon", horizon_days)

    failed = False

    # Staleness is the hard alarm — the unambiguous "collector is dead" signal.
    if age_hours > MAX_AGE_HOURS:
        log.error("STALE: newest data is %.1f h old (threshold %d h)", age_hours, MAX_AGE_HOURS)
        failed = True

    # An almost-empty horizon is the other hard alarm: the dataset is about to
    # stop accumulating days-to-departure coverage, and that can't be backfilled.
    if horizon_days is not None and horizon_days < HORIZON_HARD_DAYS:
        log.error("HORIZON: farthest anchor departs in %d days (threshold %d) — "
                  "add future DATE_PAIRS to scripts/collect.py", horizon_days, HORIZON_HARD_DAYS)
        failed = True

    if failed:
        return 1

    log.info("OK: collector is fresh")
    return 0


if __name__ == "__main__":
    sys.exit(check())