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

    age_hours = (datetime.now(timezone.utc) - newest).total_seconds() / 3600
    log.info("newest snapshot: %s (%.1f h ago), routes in that cycle: %d/%d",
             newest.isoformat(timespec="seconds"), age_hours, routes, EXPECTED_ROUTES)

    # Coverage is a soft signal: one route failing a single cycle is within
    # tolerance, so warn but don't fail on it.
    if routes < EXPECTED_ROUTES:
        log.warning("latest cycle covered only %d of %d routes", routes, EXPECTED_ROUTES)

    # Staleness is the hard alarm — the unambiguous "collector is dead" signal.
    if age_hours > MAX_AGE_HOURS:
        log.error("STALE: newest data is %.1f h old (threshold %d h)", age_hours, MAX_AGE_HOURS)
        return 1

    log.info("OK: collector is fresh")
    return 0


if __name__ == "__main__":
    sys.exit(check())