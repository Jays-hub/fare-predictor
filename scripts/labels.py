"""Forward-looking labels for the model step — built on carrier_min_series.

The mirror image of features.py: features may only look BACKWARD from a
snapshot, labels may only look FORWARD. Keeping them in separate modules makes
the boundary impossible to blur — nothing produced here may ever be fed to a
model as an input column.

The label question (framing decided 2026-06-12): standing at snapshot t of one
itinerary — "will this itinerary's carrier-min fare drop by at least
`drop_dollars` at some snapshot within the next `horizon_days` days?"

Columns added (LABEL_COLUMNS):
  - fwd_min         : min fare among this itinerary's snapshots in (t, t+H] —
                      the quantile-regression target. NaN when no snapshot
                      landed in the window.
  - fwd_n_snapshots : how many future snapshots the window contains — a
                      coverage diagnostic (blocked cycles thin it out).
  - label_observed  : True iff the window can no longer change.
  - will_drop       : 1/0 where observed, pd.NA while censored.

CENSORING IS THE CARDINAL RULE HERE. A window is "observed" only when no
future snapshot can ever land in it: either it has fully elapsed (window end
<= the panel's last snapshot — wall-clock has passed it) or the itinerary has
already departed (its collection is over). Train BOTH heads only on observed
rows.
  - Never treat an unfinished window as "no drop": that floods the panel's
    right edge with false negatives and biases the model toward "buy now".
  - Don't keep early-confirmed positives from unfinished windows either, even
    though a drop already seen cannot un-happen: keeping determined-1s while
    excluding undetermined-0s over-samples positives at that same edge.
Snapshots missed INSIDE an elapsed window (blocked cycles) are measurement
noise, not censoring — the label is defined over the fares we observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A "real" drop, mirroring the alarm's MIN_DROP_DOLLARS; H matches the ~2/day
# snapshot cadence (a 7-day window nominally holds ~14 future snapshots).
DROP_DOLLARS = 20.0
HORIZON_DAYS = 7

LABEL_COLUMNS = ["fwd_min", "fwd_n_snapshots", "label_observed", "will_drop"]


def build_labels(
    min_series: pd.DataFrame,
    drop_dollars: float = DROP_DOLLARS,
    horizon_days: int = HORIZON_DAYS,
) -> pd.DataFrame:
    """Add forward-window label columns to a carrier_min_series frame (one row
    in, one row out — same (itinerary, observed_at) grain as features.py)."""
    df = min_series.sort_values(["itinerary", "observed_at"]).reset_index(drop=True)
    horizon = pd.Timedelta(days=horizon_days)
    panel_end = df["observed_at"].max()

    fwd_min = np.full(len(df), np.nan)
    fwd_n = np.zeros(len(df), dtype=int)
    for _, g in df.groupby("itinerary", sort=False):
        idx = g.index.to_numpy()
        obs = g["observed_at"]
        prices = g["price"].to_numpy()
        # Row i's window is the half-open interval (t_i, t_i + H]; rows are
        # time-sorted, so it is exactly the slice [i+1, hi_i).
        hi = obs.searchsorted(obs + horizon, side="right")
        for i in range(len(g)):
            w = prices[i + 1 : hi[i]]
            if len(w):
                fwd_min[idx[i]] = w.min()
                fwd_n[idx[i]] = len(w)

    df["fwd_min"] = fwd_min
    df["fwd_n_snapshots"] = fwd_n

    # Observed = the window can no longer change: it has fully elapsed, or the
    # itinerary departed (collection over, no snapshot can ever be added).
    # Departure is strict (<): on the dep date itself a cycle may still run.
    elapsed = df["observed_at"] + horizon <= panel_end
    departed = df["dep_date"].dt.date < panel_end.date()
    df["label_observed"] = elapsed | departed

    # NaN fwd_min inside an observed window correctly yields 0: no lower fare
    # materialized before the window/deadline closed, so waiting caught nothing.
    drop = df["fwd_min"] <= df["price"] - drop_dollars
    df["will_drop"] = drop.astype("Int8").mask(~df["label_observed"])
    return df


if __name__ == "__main__":  # pragma: no cover
    # Real-data smoke run + Google-scorecard seed. Needs DATABASE_URL (same
    # gitignored-.env hydration as harness.py).
    import os
    from pathlib import Path

    import trajectories as T

    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists() and not os.environ.get("DATABASE_URL"):
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("DATABASE_URL"):
                os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip("'\"")

    labels = build_labels(T.carrier_min_series(T.load_snapshots()))
    n = len(labels)
    seen = labels[labels["label_observed"]]
    print(
        f"{n} rows; {len(seen)} ({len(seen) / n:.0%}) with observed "
        f"{HORIZON_DAYS}d windows (the rest are right-edge censored)."
    )
    if len(seen):
        rate = float(seen["will_drop"].mean())
        print(f"P(drop >= ${DROP_DOLLARS:.0f} within {HORIZON_DAYS}d) = {rate:.1%}")
        # Google-scorecard seed: how often did each verdict precede a real drop?
        # (Degenerate until price_level coverage deepens — zero "low" so far.)
        board = seen.groupby("price_level", dropna=False)["will_drop"].agg(["count", "mean"])
        print("\nGoogle scorecard seed (drop rate by price_level verdict):")
        print(board.to_string())
