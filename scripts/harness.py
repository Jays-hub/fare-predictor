"""Evaluation harness: judge a buy/wait policy in DOLLARS, not RMSE.

Governing principle #2 of this project: a model is only as good as the money it
saves a traveler vs. just buying now. This module is the make-or-break ruler —
build it before any model so every later baseline/model is scored the same way.

The unit of evaluation is one ITINERARY (dest x dep_date x ret_date x carrier),
exactly the row produced by trajectories.carrier_min_series(). Each itinerary is
a short price time-series across collection snapshots.

A POLICY is a causal decision rule. Walking an itinerary's snapshots in time
order, at each snapshot it is handed only the history UP TO AND INCLUDING that
snapshot and returns True ("buy now") or False ("wait"). Causality is enforced
by construction — the policy is never given a row it couldn't have seen yet. The
first True ends the simulation (you bought the ticket). If the policy waits the
whole way, it is forced to buy at the LAST observed snapshot (the deadline) —
waiting is not free.

Two reference points bracket every policy, per itinerary:
  - always-buy-now : cost = price at the FIRST snapshot. The must-beat baseline.
  - oracle         : cost = the MINIMUM price over the window (perfect timing).

Headline, dollar-weighted across itineraries:
  - $ saved vs buy-now   = buy_now_cost - policy_cost   (negative = lost money)
  - % of oracle captured = sum(saved) / sum(oracle_savings)
    Dollar-weighting (not a mean of per-itinerary ratios) is deliberate: it
    weights each itinerary by how much money was actually on the table, and it
    is well-defined even when some itineraries had zero room to save.

Time-aware evaluation: split_by_dep_date() splits the panel by DEPARTURE DATE.
Because the itinerary key contains dep_date, every snapshot of an itinerary
stays on one side of the split — snapshots of one itinerary are never split
across train/test (the cardinal sin for this panel).
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

# A policy sees the history-so-far (rows of ONE itinerary's carrier_min_series,
# sorted by observed_at, up to and including "now" = the last row) and decides.
Policy = Callable[[pd.DataFrame], bool]


# ------------------------------------------------------------------ core engine

def simulate(min_series: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    """Run `policy` over every itinerary; return one outcome row per itinerary.

    Input is a carrier_min_series frame (or any frame with columns: itinerary,
    observed_at, price, days_to_dep, dest, dep_date, carrier). Output columns:
    buy/oracle/policy prices, dollars saved, oracle headroom, and the
    days-to-departure at which the policy actually bought.
    """
    rows = []
    for itin, g in min_series.groupby("itinerary", sort=False):
        g = g.sort_values("observed_at").reset_index(drop=True)
        prices = g["price"].to_numpy()

        buy_i = len(g) - 1  # default: never triggered -> forced deadline buy
        for i in range(len(g)):
            if bool(policy(g.iloc[: i + 1])):
                buy_i = i
                break

        buy_now = float(prices[0])
        oracle = float(prices.min())
        policy_price = float(prices[buy_i])
        rows.append(
            {
                "itinerary": itin,
                "dest": g["dest"].iloc[0],
                "dep_date": g["dep_date"].iloc[0],
                "carrier": g["carrier"].iloc[0],
                "n_snapshots": len(g),
                "buy_now_price": buy_now,
                "oracle_price": oracle,
                "policy_price": policy_price,
                "buy_days_to_dep": int(g["days_to_dep"].iloc[buy_i]),
                "saved_vs_buynow": buy_now - policy_price,
                "oracle_savings": buy_now - oracle,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Per-itinerary capture ratio is only defined when there was room to save;
    # the dollar-weighted aggregate in summarize() is the headline number.
    out["pct_oracle"] = pd.NA
    has_room = out["oracle_savings"] > 0
    out.loc[has_room, "pct_oracle"] = (
        out.loc[has_room, "saved_vs_buynow"] / out.loc[has_room, "oracle_savings"]
    )
    return out


def summarize(outcomes: pd.DataFrame) -> dict:
    """Collapse per-itinerary outcomes into the headline dollar metrics."""
    if outcomes.empty:
        return {"n_itineraries": 0}
    tot_saved = float(outcomes["saved_vs_buynow"].sum())
    tot_oracle = float(outcomes["oracle_savings"].sum())
    return {
        "n_itineraries": int(len(outcomes)),
        "total_saved_vs_buynow": tot_saved,
        "total_oracle_savings": tot_oracle,
        "pct_oracle_captured": (tot_saved / tot_oracle) if tot_oracle else float("nan"),
        "mean_saved_per_itin": float(outcomes["saved_vs_buynow"].mean()),
        "n_beat_buynow": int((outcomes["saved_vs_buynow"] > 0).sum()),
        "n_lost_to_buynow": int((outcomes["saved_vs_buynow"] < 0).sum()),
    }


def evaluate(min_series: pd.DataFrame, policy: Policy) -> dict:
    """Convenience: simulate then summarize."""
    return summarize(simulate(min_series, policy))


# -------------------------------------------------------------- time-aware split

def split_by_dep_date(min_series: pd.DataFrame, cutoff) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by departure date: train = dep_date < cutoff, test = dep_date >= cutoff.

    Itineraries (and therefore all their snapshots) land wholly on one side.
    """
    cutoff = pd.Timestamp(cutoff)
    train = min_series[min_series["dep_date"] < cutoff].copy()
    test = min_series[min_series["dep_date"] >= cutoff].copy()
    return train, test


# ------------------------------------------------------------- baseline policies
# Naive, causal reference rules. The first real model must beat these in dollars.

def always_buy_now(history: pd.DataFrame) -> bool:
    """Buy at the first snapshot. The must-beat baseline; saves $0 by definition."""
    return True


def never_buy(history: pd.DataFrame) -> bool:
    """Always wait -> forced to buy at the deadline (last snapshot)."""
    return False


def fixed_days_out(n: int) -> Policy:
    """Buy as soon as the itinerary is within `n` days of departure."""

    def policy(history: pd.DataFrame) -> bool:
        return int(history["days_to_dep"].iloc[-1]) <= n

    return policy


def buy_iff_google_low(history: pd.DataFrame) -> bool:
    """Buy the moment Google's own verdict for this query says the fare is "low".

    This is the must-beat reference baseline from principle #2: if a model can't
    beat "just trust Google's low/typical/high label," it isn't earning its keep.
    price_level is nullable (un-backfillable before it shipped 2026-06-10) and no
    "low" has been observed yet, so on the current panel this never fires and
    degrades to a deadline buy — an all-"wait" result here is expected, not a bug.
    """
    pl = history["price_level"].iloc[-1]
    return isinstance(pl, str) and pl.lower() == "low"


def buy_below_trailing_median(frac: float = 1.0, min_history: int = 2) -> Policy:
    """Buy when the current fare is <= `frac` x the median of all PRIOR snapshots.

    Needs at least `min_history` observations before it can fire; until then it
    waits. A dip-buyer baseline (frac<1 demands a real discount below trend).
    """

    def policy(history: pd.DataFrame) -> bool:
        if len(history) < min_history:
            return False
        prior = history["price"].iloc[:-1]
        return float(history["price"].iloc[-1]) <= frac * float(prior.median())

    return policy


# Registry for the CLI runner / quick comparisons.
BASELINES: dict[str, Policy] = {
    "always_buy_now": always_buy_now,
    "deadline_buy": never_buy,
    "fixed_21d_out": fixed_days_out(21),
    "fixed_14d_out": fixed_days_out(14),
    "below_trailing_median": buy_below_trailing_median(1.0),
    "dip_5pct_below_median": buy_below_trailing_median(0.95),
    "google_says_low": buy_iff_google_low,
}


# ----------------------------------------------------------------- CLI / real run

def _compare(min_series: pd.DataFrame, policies: dict[str, Policy]) -> pd.DataFrame:
    """One summary row per policy — the at-a-glance leaderboard."""
    return pd.DataFrame(
        {name: summarize(simulate(min_series, p)) for name, p in policies.items()}
    ).T


if __name__ == "__main__":  # pragma: no cover
    # Real-data smoke run: needs DATABASE_URL (loaded from a gitignored .env the
    # same way the EDA script reads it). Prints the baseline leaderboard.
    import os
    from pathlib import Path

    import trajectories as T

    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists() and not os.environ.get("DATABASE_URL"):
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.startswith("DATABASE_URL"):
                os.environ["DATABASE_URL"] = line.split("=", 1)[1].strip().strip("'\"")

    df = T.load_snapshots()
    mins = T.carrier_min_series(df)
    print(f"Loaded {len(df)} rows -> {mins['itinerary'].nunique()} itineraries, "
          f"{mins['observed_at'].nunique()} snapshots.\n")
    board = _compare(mins, BASELINES)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(board.to_string())
