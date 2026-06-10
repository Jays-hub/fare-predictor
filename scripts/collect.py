import csv
import logging
import random
import re
import time
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from fast_flights import FlightData, Passengers, Result, get_flights

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("collector")


def _is_usable(result: Result) -> bool:
    """At least one flight must carry a name. Google sometimes returns a
    stripped layout where prices parse but names/times come back empty —
    those rows can't be attributed to a carrier, so the whole response is
    a soft failure we should retry rather than store."""
    return any(f.name for f in result.flights)

# ---- Configuration -------------------------------------------------------

ORIGIN = "ATL"
CARRIERS = {"Delta", "Frontier"}            # everything else is dropped at parse time

# Destinations served from ATL by both carriers (Frontier is the limiting set).
# Start small; expand toward the master-plan list (DEN, LAS, MCO, TPA, ...).
ROUTES = ["MCO", "LAS", "DEN"]

# Trip shapes as FIXED calendar (dep, ret) pairs. These are anchors observed
# every run — that repetition is what produces price trajectories. Do NOT
# regenerate them from `today` each run (see build_date_pairs for why).
#
# Ladder policy: keep 3-5 staggered anchors per trip shape so there is always
# a spread of days-to-departure in the panel. Expired anchors are skipped
# automatically at run time; the freshness monitor emails when the horizon
# drops below ~2 weeks. When adding anchors, keep the day-of-week shapes
# consistent (3-day = Fri->Mon, 7-day = Mon->Mon) so trajectories stay
# comparable across anchors.
DATE_PAIRS = [
    ("2026-06-26", "2026-06-29"),           # 3-day weekend (Fri->Mon)
    ("2026-07-10", "2026-07-13"),           # 3-day weekend (Fri->Mon)
    ("2026-07-20", "2026-07-27"),           # 7-day trip (Mon->Mon)
    ("2026-07-31", "2026-08-03"),           # 3-day weekend (Fri->Mon)
    ("2026-08-17", "2026-08-24"),           # 7-day trip (Mon->Mon)
]

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = REPO_ROOT / "data" / "raw" / "snapshots.csv"

FIELDNAMES = [
    "observed_at", "origin", "dest", "dep_date", "ret_date",
    "carrier", "cabin", "price", "stops", "nonstop", "duration_min",
    "dep_time_raw", "dep_hour", "dep_minute", "dep_dow", "source",
    "price_level",
]


def build_date_pairs(anchor: date, specs: list[tuple[int, int]]) -> list[tuple[str, str]]:
    """Generate (dep, ret) pairs from (days_out, trip_length) specs relative to
    a FIXED anchor. Run once to seed DATE_PAIRS, then paste the result in as
    constants. Anchoring to a fixed date — not today — keeps the calendar dates
    stable across runs so trajectories accumulate instead of drifting daily."""
    pairs = []
    for days_out, length in specs:
        dep = anchor + timedelta(days=days_out)
        ret = dep + timedelta(days=length)
        pairs.append((dep.isoformat(), ret.isoformat()))
    return pairs


def live_date_pairs(pairs: list[tuple[str, str]], today_iso: str) -> list[tuple[str, str]]:
    """Date-pairs whose departure hasn't passed. Expired anchors otherwise burn
    a full retry budget per route on queries that can no longer succeed."""
    return [(dep, ret) for dep, ret in pairs if dep >= today_iso]


def fetch_with_retry(
    flight_data: list[FlightData],
    *,
    max_attempts: int = 4,
    base_delay: float = 3.0,
) -> Result | None:
    """Fetch a round-trip query, surviving the intermittent blocking and the
    degraded-parse case. Exponential backoff with jitter between attempts.
    Returns a usable Result, or None if every attempt fails."""
    for attempt in range(1, max_attempts + 1):
        try:
            result = get_flights(
                flight_data=flight_data,
                trip="round-trip",
                seat="economy",
                passengers=Passengers(adults=1),
            )
        except Exception as e:
            # RuntimeError = the "Loading results" stub; AssertionError = non-200.
            # Truncate because the RuntimeError message embeds the whole page dump.
            log.warning("attempt %d/%d raised %s: %s",
                        attempt, max_attempts, type(e).__name__, str(e)[:120])
            result = None

        if result is not None and _is_usable(result):
            log.info("attempt %d/%d ok (%d flights)", attempt, max_attempts, len(result.flights))
            return result
        if result is not None:
            log.warning("attempt %d/%d degraded parse (no names)", attempt, max_attempts)

        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1.5)
            log.info("backing off %.1fs", delay)
            time.sleep(delay)

    log.error("all %d attempts failed", max_attempts)
    return None


def _parse_price(raw: str) -> int | None:
    """'$1,234' -> 1234. Returns None for missing/zero/garbage prices."""
    digits = raw.replace("$", "").replace(",", "").strip()
    if not digits.isdigit():
        return None
    value = int(digits)
    return value if value > 0 else None


def _parse_departure(raw: str) -> dict:
    """'5:05 AM on Thu, Jun 18' -> structured time-of-day fields.
    Stores precise values; buckets like 'redeye' are derived at modeling time
    so their definitions can change without recollecting."""
    out = {"dep_time_raw": raw, "dep_hour": None, "dep_minute": None, "dep_dow": None}
    if not raw:
        return out

    m = re.search(r"(\d{1,2}):(\d{2})\s*(AM|PM)", raw)
    if m:
        hour = int(m.group(1)) % 12
        if m.group(3) == "PM":
            hour += 12
        out["dep_hour"] = hour            # 0-23, the key field for time-of-day modeling
        out["dep_minute"] = int(m.group(2))

    d = re.search(r"on\s+(\w{3}),", raw)  # 'Thu' -> day-of-week
    if d:
        out["dep_dow"] = d.group(1)

    return out


def _duration_to_minutes(raw: str) -> int | None:
    """'1 hr 40 min' -> 100. Stored as an int so it's model-ready."""
    if not raw:
        return None
    hours = re.search(r"(\d+)\s*hr", raw)
    mins = re.search(r"(\d+)\s*min", raw)
    if not hours and not mins:
        return None
    return (int(hours.group(1)) * 60 if hours else 0) + (int(mins.group(1)) if mins else 0)


def parse_result(
    result: Result,
    *,
    origin: str,
    dest: str,
    dep_date: str,
    ret_date: str,
    observed_at: str | None = None,
    carriers: set[str] | None = None,
    cabin: str = "economy",
    source: str = "google_flights/fast-flights",
) -> list[dict]:
    """Turn one usable round-trip Result into clean storage rows.

    Dedups the best-listed-twice quirk, filters to the carriers we track,
    parses price/duration, and stamps each row with query metadata + a single
    shared observed_at. Price is the round-trip TOTAL for that outbound option
    (confirmed against the live UI)."""
    if observed_at is None:
        observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if carriers is None:
        carriers = {"Delta", "Frontier"}

    # Google's own low/typical/high verdict for this query — one value per
    # Result, denormalized onto every row. It can't be backfilled, and it is
    # both a must-beat baseline ("buy iff Google says low") and a feature.
    price_level = (result.current_price or "").strip().lower() or None

    rows: list[dict] = []
    seen: set[tuple] = set()

    for f in result.flights:
        if f.name not in carriers:        # drops Southwest/American/etc + nameless rows
            continue

        # Dedup: the same flight appears in both the "best" and "all" sections.
        key = (f.name, f.departure, f.arrival, f.duration, f.stops, f.price)
        if key in seen:
            continue
        seen.add(key)

        price = _parse_price(f.price)
        if price is None:                 # a row we can't price is useless
            continue

        stops = f.stops if isinstance(f.stops, int) else None
        nonstop = (stops == 0) if stops is not None else None

        dep = _parse_departure(f.departure)
        rows.append({
            "observed_at": observed_at,
            "origin": origin,
            "dest": dest,
            "dep_date": dep_date,
            "ret_date": ret_date,
            "carrier": f.name,
            "cabin": cabin,
            "price": price,
            "stops": stops,
            "nonstop": nonstop,
            "duration_min": _duration_to_minutes(f.duration),
            **dep,                         # dep_time_raw, dep_hour, dep_minute, dep_dow
            "source": source,
            "price_level": price_level,
        })

    return rows


def save_rows(rows: list[dict], path: Path = DATA_FILE) -> None:
    """Append rows to the CSV, writing the header only when the file is new.
    Append-only: we never overwrite, because the accumulating history is the dataset."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


try:
    import psycopg
except ImportError:
    psycopg = None

_INSERT_SQL = """
INSERT INTO snapshots (
    observed_at, origin, dest, dep_date, ret_date, carrier, cabin,
    price, stops, nonstop, duration_min, dep_time_raw, dep_hour,
    dep_minute, dep_dow, source, price_level
) VALUES (
    %(observed_at)s, %(origin)s, %(dest)s, %(dep_date)s, %(ret_date)s,
    %(carrier)s, %(cabin)s, %(price)s, %(stops)s, %(nonstop)s,
    %(duration_min)s, %(dep_time_raw)s, %(dep_hour)s, %(dep_minute)s,
    %(dep_dow)s, %(source)s, %(price_level)s
)
ON CONFLICT DO NOTHING
"""

# Idempotent micro-migration (price_level added 2026-06-10). Running it inline
# means the scheduled runner can never race a manual ALTER and drop a cycle's
# rows — there is no window where the INSERT references a missing column.
# Canonical DDL lives in db/schema.sql.
_ENSURE_SCHEMA_SQL = "ALTER TABLE snapshots ADD COLUMN IF NOT EXISTS price_level text"


def save_to_postgres(rows: list[dict]) -> int:
    """Write rows to Postgres when DATABASE_URL is set. No-op otherwise, so
    local dev needs zero config. Failures are logged, not raised — a DB hiccup
    on one combo shouldn't crash the cycle; the next run recovers it."""
    dsn = os.environ.get("DATABASE_URL")
    if not rows:
        return 0
    if not dsn:
        log.info("DATABASE_URL not set; CSV only (expected for local dev)")
        return 0
    if psycopg is None:
        log.warning("DATABASE_URL set but psycopg not installed; skipping DB write")
        return 0
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(_ENSURE_SCHEMA_SQL)
                cur.executemany(_INSERT_SQL, rows)
                # psycopg3 accumulates affected rows across executemany, so this
                # is the real insert count — ON CONFLICT skips don't inflate it.
                inserted = cur.rowcount if cur.rowcount >= 0 else len(rows)
            conn.commit()                         # explicit; no-op if context already commits
        log.info("postgres: wrote %d rows (%d duplicate(s) skipped)",
                 inserted, len(rows) - inserted)
        return inserted
    except Exception as e:
        log.error("postgres write failed (%s); rows kept in CSV only", type(e).__name__)
        return 0


def persist(rows: list[dict]) -> None:
    """Always write CSV (local inspectability); also write Postgres when
    DATABASE_URL is present (scheduled runs)."""
    save_rows(rows)
    save_to_postgres(rows)


def _collect_one(dest: str, dep_date: str, ret_date: str, observed_at: str) -> int | None:
    """Fetch + parse + save one route x date-pair. Returns rows stored, or None on failure."""
    flight_data = [
        FlightData(date=dep_date, from_airport=ORIGIN, to_airport=dest),
        FlightData(date=ret_date, from_airport=dest, to_airport=ORIGIN),
    ]
    result = fetch_with_retry(flight_data)
    if result is None:
        return None
    rows = parse_result(
        result, origin=ORIGIN, dest=dest,
        dep_date=dep_date, ret_date=ret_date,
        observed_at=observed_at, carriers=CARRIERS,
    )
    persist(rows)
    return len(rows)


def run_once() -> None:
    """One cycle. First pass over every combo; failures get a single second
    pass after a longer cooldown to decorrelate from transient blocking."""
    observed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # UTC date, consistent with the days_to_dep convention in trajectories.py.
    today = datetime.now(timezone.utc).date().isoformat()
    pairs = live_date_pairs(DATE_PAIRS, today)
    for dep, ret in DATE_PAIRS:
        if (dep, ret) not in pairs:
            log.warning("skipping expired date-pair %s/%s (departed)", dep, ret)

    combos = [(dest, dep, ret) for dest in ROUTES for dep, ret in pairs]
    total = 0

    failed = []
    for dest, dep_date, ret_date in combos:
        log.info("querying %s->%s %s/%s", ORIGIN, dest, dep_date, ret_date)
        n = _collect_one(dest, dep_date, ret_date, observed_at)
        if n is None:
            failed.append((dest, dep_date, ret_date))
            log.warning("first pass failed: %s->%s %s/%s", ORIGIN, dest, dep_date, ret_date)
        else:
            total += n
            log.info("stored %d rows for %s->%s %s", n, ORIGIN, dest, dep_date)
        time.sleep(random.uniform(2.0, 5.0))

    unrecoverable = 0
    if failed:
        cooldown = random.uniform(30, 60)
        log.info("%d combo(s) failed; cooling %.0fs before second pass", len(failed), cooldown)
        time.sleep(cooldown)
        for dest, dep_date, ret_date in failed:
            n = _collect_one(dest, dep_date, ret_date, observed_at)
            if n is None:
                unrecoverable += 1
                log.warning("dropped after second pass: %s->%s %s/%s", ORIGIN, dest, dep_date, ret_date)
            else:
                total += n
                log.info("recovered: %s->%s %s/%s (%d rows)", ORIGIN, dest, dep_date, ret_date, n)
            time.sleep(random.uniform(2.0, 5.0))

    log.info("cycle complete: %d rows, %d combo(s) unrecoverable", total, unrecoverable)

if __name__ == "__main__":
    run_once()