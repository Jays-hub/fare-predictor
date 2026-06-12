"""Mistake-fare alarm: flag fares that fall well below their recent trailing min.

This is the fastest real-money win in the roadmap — it needs almost no data and
no model, just an outlier rule against a per-itinerary trailing baseline.

It deliberately reuses the proven fail->email channel (like check_freshness.py),
but with INVERTED semantics:

    exit 1  ==  a deal was found      -> GitHub marks the run "failed" -> email
    exit 0  ==  nothing unusual       -> silence is the normal, healthy state

So an email from THIS job is good news (go buy), whereas an email from the
freshness monitor is bad news. Same plumbing, opposite meaning — documented here
so future-me doesn't "fix" the non-zero exit.

Detection (per buy-vs-wait unit = carrier_min_series itinerary):
  - baseline   = the MINIMUM cheapest-fare over the trailing WINDOW_DAYS, taken
                 over snapshots STRICTLY BEFORE the latest cycle.
  - a deal     = latest fare is both >= DROP_THRESHOLD below that baseline AND a
                 drop of at least MIN_DROP_DOLLARS (kills noise on cheap fares).
  - guardrail  = require at least MIN_PRIOR prior snapshots, else there is no
                 trustworthy baseline yet and we stay quiet (no false alarm).

Only the LATEST cycle is judged — we alert on a live, buyable price, never on a
deal that already evaporated in the historical panel.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deals")

# A 14-day trailing window matches the project's stated baseline and is long
# enough to establish a normal price band on a panel sampled ~2x/day.
WINDOW_DAYS = 14
# A fare must beat its recent trailing min by this fraction to count as a deal.
# Set high enough that ordinary day-to-day wiggle never trips it.
DROP_THRESHOLD = 0.20
# ...and by at least this many dollars, so a 20% dip on a $60 fare ($12) doesn't
# spam — a mistake fare worth a special trip is a real-dollars event.
MIN_DROP_DOLLARS = 20
# Without this much prior history there is no trustworthy baseline; stay silent.
MIN_PRIOR = 3


def find_deals(
    min_series: pd.DataFrame,
    window_days: int = WINDOW_DAYS,
    drop_threshold: float = DROP_THRESHOLD,
    min_drop_dollars: float = MIN_DROP_DOLLARS,
    min_prior: int = MIN_PRIOR,
) -> pd.DataFrame:
    """Pure detector over a carrier_min_series frame. No IO — unit-testable.

    Returns one row per flagged itinerary (empty frame if nothing qualifies),
    sorted by the biggest percentage drop first.
    """
    if min_series.empty:
        return min_series.iloc[0:0]

    latest = min_series["observed_at"].max()
    cutoff = latest - pd.Timedelta(days=window_days)
    latest_rows = min_series[min_series["observed_at"] == latest]

    deals = []
    for _, now in latest_rows.iterrows():
        itin = now["itinerary"]
        hist = min_series[
            (min_series["itinerary"] == itin)
            & (min_series["observed_at"] < latest)
            & (min_series["observed_at"] >= cutoff)
        ]
        if len(hist) < min_prior:
            continue  # no trustworthy baseline yet -> stay quiet

        baseline = float(hist["price"].min())
        latest_price = float(now["price"])
        drop = baseline - latest_price
        if drop < min_drop_dollars:
            continue
        drop_pct = drop / baseline
        if drop_pct < drop_threshold:
            continue

        deals.append(
            {
                "itinerary": itin,
                "dest": now["dest"],
                "carrier": now["carrier"],
                "dep_date": now["dep_date"],
                "latest_price": latest_price,
                "trailing_min": baseline,
                "drop_dollars": drop,
                "drop_pct": drop_pct,
                "n_prior": len(hist),
            }
        )

    out = pd.DataFrame(deals)
    if out.empty:
        return out
    return out.sort_values("drop_pct", ascending=False).reset_index(drop=True)


def check() -> int:
    """Load the panel, run the detector, and translate the result into an exit
    code: 1 (=> email) when a deal is live, 0 when all quiet."""
    # Local convenience: hydrate DATABASE_URL from a gitignored .env, same as the
    # EDA/harness scripts. In CI the secret is already in the environment.
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists() and not os.environ.get("DATABASE_URL"):
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("DATABASE_URL") and "=" in line:
                os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip("'\"")

    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL not set; cannot check for deals")
        return 1  # a config failure should still surface, not pass silently

    # Imported here so the pure detector stays importable without DB deps.
    import trajectories as T

    df = T.load_snapshots()
    mins = T.carrier_min_series(df)
    if mins.empty:
        log.info("no snapshots yet; nothing to check")
        return 0

    latest = mins["observed_at"].max()
    deals = find_deals(mins)

    log.info(
        "checked %d itineraries in latest cycle %s; %d deal(s) found "
        "(window %dd, threshold %.0f%% / $%d)",
        (mins["observed_at"] == latest).sum(),
        latest.isoformat(timespec="seconds"),
        len(deals),
        WINDOW_DAYS,
        DROP_THRESHOLD * 100,
        MIN_DROP_DOLLARS,
    )

    if deals.empty:
        log.info("OK: no mistake fares")
        return 0

    for _, d in deals.iterrows():
        log.warning(
            "DEAL: %s %s dep %s — $%.0f, down %.0f%% ($%.0f) from %dd min $%.0f",
            d["dest"], d["carrier"], pd.Timestamp(d["dep_date"]).date(),
            d["latest_price"], d["drop_pct"] * 100, d["drop_dollars"],
            WINDOW_DAYS, d["trailing_min"],
        )
    return 1  # deal(s) live -> non-zero -> GitHub failure email


if __name__ == "__main__":
    sys.exit(check())
