"""Trajectory reconstruction: turn raw fare snapshots into modeling-ready series.

IO is separated from transformation on purpose:
  - load_snapshots() is the only function that touches Neon.
  - every other function takes a DataFrame and returns a DataFrame, so the
    reshaping logic is testable with synthetic data (no DB, no network) and
    reusable by both notebooks and the eval harness.

Two parallel views are produced from the same raw rows:
  - carrier_min_series(): the per-carrier CHEAPEST fare per snapshot. This is
    the headline buy-vs-wait signal; it sidesteps flight-identity fragility.
  - departure_series(): every individual departure tracked over time, keyed by
    dep_hour/dep_minute, for time-of-day modeling ("is the redeye cheapest?").
"""

import os

import pandas as pd

# Collection runs at 10:00 and 22:00 UTC = 06:00 and 18:00 ET. At BOTH of those
# instants the UTC calendar date equals the Eastern calendar date (the two only
# diverge between 00:00-05:00 UTC, which we never sample). So days-to-departure
# computed on the UTC date is exact for scheduled data, with no tzdata dependency.
# (Ad-hoc manual runs in the 00:00-05:00 UTC window could be off by one day;
# acceptable, and flagged here so it isn't a mystery later.)

_COLUMNS = (
    "observed_at, origin, dest, dep_date, ret_date, carrier, cabin, price, "
    "stops, nonstop, duration_min, dep_hour, dep_minute, dep_dow"
)


# --------------------------------------------------------------------------- IO

def load_snapshots(dsn: str | None = None) -> pd.DataFrame:
    """Pull all snapshot rows from Neon into a typed, sorted DataFrame.

    The only DB-touching function. Everything downstream operates on the frame
    it returns, so it can be swapped for a CSV/Parquet loader without touching
    any reshaping code.
    """
    import psycopg  # imported here so the transform functions don't require it

    dsn = dsn or os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL not set and no dsn passed")

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT {_COLUMNS} FROM snapshots")
            cols = [d.name for d in cur.description]
            rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=cols)
    return _coerce_types(df)


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes coming from the DB (or a synthetic frame) so the rest of
    the module can assume tz-aware observed_at and tz-naive trip dates."""
    df = df.copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    df["dep_date"] = pd.to_datetime(df["dep_date"]).dt.tz_localize(None).dt.normalize()
    df["ret_date"] = pd.to_datetime(df["ret_date"]).dt.tz_localize(None).dt.normalize()
    for c in ("price", "stops", "duration_min", "dep_hour", "dep_minute"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values(["dest", "dep_date", "ret_date", "carrier", "observed_at"]).reset_index(drop=True)


# -------------------------------------------------------------- derived helpers

def _days_to_dep(dep_date: pd.Series, observed_at: pd.Series) -> pd.Series:
    """Whole days from the observation date to the departure date.

    Uses the UTC calendar date of observed_at, which equals the Eastern date at
    the scheduled 10:00/22:00 UTC snapshot times (see note at top of module).
    """
    observed_day = observed_at.dt.tz_localize(None).dt.normalize()
    return (dep_date - observed_day).dt.days


def itinerary_id(df: pd.DataFrame) -> pd.Series:
    """Stable string identifying one (route x date-pair x carrier) itinerary."""
    return (
        df["dest"] + "|" + df["dep_date"].dt.strftime("%Y-%m-%d")
        + "|" + df["ret_date"].dt.strftime("%Y-%m-%d") + "|" + df["carrier"]
    )


# ------------------------------------------------------------- trajectory views

def carrier_min_series(df: pd.DataFrame) -> pd.DataFrame:
    """Headline series: cheapest fare per (itinerary, snapshot).

    One row per (dest, dep_date, ret_date, carrier, observed_at). This is the
    clean buy-vs-wait unit and the input to the eval harness.
    """
    keys = ["dest", "dep_date", "ret_date", "carrier", "observed_at"]
    out = df.groupby(keys, as_index=False).agg(
        price=("price", "min"),
        n_flights=("price", "size"),
        nonstop_available=("nonstop", "max"),
    )
    out["days_to_dep"] = _days_to_dep(out["dep_date"], out["observed_at"])
    out["itinerary"] = itinerary_id(out)
    return out.sort_values(["itinerary", "observed_at"]).reset_index(drop=True)


def departure_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-departure view for time-of-day modeling.

    Each row stays an individual departure; we add days_to_dep and a flight_key
    that identifies the same departure across snapshots, so a single redeye can
    be tracked over time.
    """
    out = df.copy()
    out["days_to_dep"] = _days_to_dep(out["dep_date"], out["observed_at"])
    out["itinerary"] = itinerary_id(out)
    hh = out["dep_hour"].astype("Int64").astype(str).str.zfill(2)
    mm = out["dep_minute"].astype("Int64").astype(str).str.zfill(2)
    out["flight_key"] = out["itinerary"] + "@" + hh + ":" + mm
    return out.sort_values(["flight_key", "observed_at"]).reset_index(drop=True)


def pivot_min_by_carrier(min_series: pd.DataFrame, dest: str, dep_date: str, ret_date: str) -> pd.DataFrame:
    """Wide table for one route x date-pair: observed_at index, one column per
    carrier's cheapest fare. Convenient for co-movement plots and correlations.
    """
    m = min_series[
        (min_series["dest"] == dest)
        & (min_series["dep_date"] == pd.Timestamp(dep_date))
        & (min_series["ret_date"] == pd.Timestamp(ret_date))
    ]
    return m.pivot_table(index="observed_at", columns="carrier", values="price").sort_index()
