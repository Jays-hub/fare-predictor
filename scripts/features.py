"""Feature engineering for the buy-vs-wait model — built on carrier_min_series.

Separated from the (future) model the same way trajectories is separated from
the harness: build_features() takes a DataFrame and returns a DataFrame, so the
whole feature layer is testable on synthetic data and the model step becomes a
thin wrapper (fit/predict) over these columns.

THE CARDINAL RULE HERE IS CAUSALITY. Every feature is computable from
information available AT OR BEFORE its snapshot — lags/rolling/expanding use
only past+present rows, and the cross-carrier feature uses the competitor's
SAME-snapshot price (the live market, observable now). Nothing peeks forward.
That is what lets the harness score a model on these features without leaking
the future across its time-aware split. Labels (which DO look forward) are
deliberately NOT built here — they belong to the model step.

Feature families:
  - own-trajectory: lag, delta, pct-change, trailing min, distance above it,
    rolling min/mean/std, history depth.
  - fare-steps (thesis C): up-step flag/count and last up-step size.
  - min persistence: days_min_held = snapshots since this itinerary last set a
    new trailing minimum.
  - cross-carrier (THE EDGE, principle #3): the competing carrier's cheapest
    same-snapshot fare on the identical route x date-pair, and this fare's
    gap/ratio to it. This is the Delta<->Frontier co-movement signal the whole
    project is scoped around.

Holiday flags are intentionally deferred — they key off dep_date, not the
trajectory, so they belong with other calendar features in the model step.
"""

from __future__ import annotations

import pandas as pd

# Route x date-pair x snapshot identifies one competitive market; the carriers
# in it price against each other. (Adding carrier makes it the row key.)
_MARKET = ["dest", "dep_date", "ret_date", "observed_at"]


def build_features(min_series: pd.DataFrame, windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Add causal model features to a carrier_min_series frame (one row in, one
    row out — same (itinerary, observed_at) grain)."""
    df = min_series.sort_values(["itinerary", "observed_at"]).reset_index(drop=True)
    g = df.groupby("itinerary", sort=False)

    # --- own-trajectory ----------------------------------------------------
    df["price_lag1"] = g["price"].shift(1)
    df["price_delta1"] = df["price"] - df["price_lag1"]
    df["pct_change1"] = df["price_delta1"] / df["price_lag1"]
    df["trailing_min"] = g["price"].cummin()
    df["vs_trailing_min"] = df["price"] - df["trailing_min"]
    df["snapshots_seen"] = g.cumcount() + 1

    for w in windows:
        roll = g["price"].transform(lambda s, w=w: s.rolling(w, min_periods=1).min())
        df[f"roll_min_{w}"] = roll
        df[f"roll_mean_{w}"] = g["price"].transform(lambda s, w=w: s.rolling(w, min_periods=1).mean())
        df[f"roll_std_{w}"] = g["price"].transform(lambda s, w=w: s.rolling(w, min_periods=1).std())

    # --- fare-steps (thesis C) --------------------------------------------
    df["is_up_step"] = (df["price_delta1"] > 0).astype(int)
    df["n_up_steps"] = g["is_up_step"].cumsum()
    up_size = df["price_delta1"].where(df["price_delta1"] > 0)
    df["last_up_step"] = up_size.groupby(df["itinerary"]).ffill()

    # --- min persistence ---------------------------------------------------
    # A new minimum resets the counter: segment the series at each point where
    # the expanding min drops (and at the first row), then count within segment.
    prev_tmin = df.groupby("itinerary")["trailing_min"].shift(1)
    new_min = df["trailing_min"].ne(prev_tmin)              # True at group start + each new low
    seg = new_min.groupby(df["itinerary"]).cumsum()
    df["days_min_held"] = df.groupby([df["itinerary"], seg]).cumcount() + 1

    # --- cross-carrier (the edge) -----------------------------------------
    df = _add_competitor_price(df)
    df["price_vs_competitor"] = df["price"] - df["competitor_price"]
    df["price_ratio_competitor"] = df["price"] / df["competitor_price"]

    return df


def _add_competitor_price(df: pd.DataFrame) -> pd.DataFrame:
    """Attach, per row, the cheapest SAME-snapshot fare from the OTHER carrier(s)
    on the identical route x date-pair. NaN when no competitor was observed that
    snapshot (e.g. the competitor's query was blocked). Same-time data only, so
    this stays causal."""
    self_join = df.merge(
        df[_MARKET + ["carrier", "price"]],
        on=_MARKET,
        suffixes=("", "_other"),
    )
    others = self_join[self_join["carrier"] != self_join["carrier_other"]]
    competitor = (
        others.groupby(_MARKET + ["carrier"])["price_other"].min().rename("competitor_price")
    )
    # Left-merge on the row key (route x date-pair x snapshot x carrier is unique
    # in carrier_min_series), so row count and order are preserved.
    return df.merge(competitor, on=_MARKET + ["carrier"], how="left")


# Columns build_features adds, handy for the model step to select on.
FEATURE_COLUMNS = [
    "price_lag1", "price_delta1", "pct_change1", "trailing_min", "vs_trailing_min",
    "snapshots_seen", "roll_min_3", "roll_mean_3", "roll_std_3",
    "roll_min_5", "roll_mean_5", "roll_std_5",
    "is_up_step", "n_up_steps", "last_up_step", "days_min_held",
    "competitor_price", "price_vs_competitor", "price_ratio_competitor",
    "days_to_dep", "price_level",
]
