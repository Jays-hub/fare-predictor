# %% [markdown]
# # ATL fare predictor — thesis-validation EDA
#
# Purpose: on ~5 days of real data, check whether the project's core assumptions
# actually hold *before* building features and models that depend on them.
# This is intentionally directional, not conclusive — with ~10 snapshots per
# itinerary, treat everything here as "is the signal plausibly there?", not proof.
#
# Three theses under test:
#   A. Do prices on these itineraries move at all? (flat => premise is weak)
#   B. Do Delta and Frontier co-move on shared routes? (your signature feature)
#   C. Are there upward steps in the lowest fare? (free bucket-depletion proxy)
#
# Run cell-by-cell in VSCode's interactive window (Shift+Enter). Requires:
#   pip install pandas matplotlib psycopg[binary]
# and DATABASE_URL set in the environment you launch VSCode from.

# %%
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Make scripts/trajectories.py importable regardless of where the interactive
# window's cwd is. Walk up until we find the repo root (the dir containing scripts/).
here = Path.cwd()
for cand in [here, *here.parents]:
    if (cand / "scripts" / "trajectories.py").exists():
        sys.path.insert(0, str(cand / "scripts"))
        break
else:
    raise RuntimeError("could not locate scripts/trajectories.py — run from inside the repo")

import trajectories as T  # noqa: E402

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

# %% [markdown]
# ## Load + sanity check
# Confirm the data is shaped the way we think before drawing any conclusions.

# %%
df = T.load_snapshots()
print(f"rows: {len(df):,}")
print(f"observed_at range: {df.observed_at.min()}  ->  {df.observed_at.max()}")
print(f"distinct snapshots: {df.observed_at.nunique()}")
print(f"routes: {sorted(df.dest.unique())}")
print(f"carriers: {sorted(df.carrier.unique())}")
print(f"date-pairs: {sorted(set(zip(df.dep_date.dt.date, df.ret_date.dt.date)))}")
print("\nnulls per column:")
print(df.isna().sum().to_string())

# %% [markdown]
# ### Snapshots per itinerary
# How many observations do we actually have per (route x date-pair x carrier)?
# This caps what any trajectory analysis can show — singletons can't reveal movement.

# %%
mins = T.carrier_min_series(df)
coverage = (
    mins.groupby("itinerary")
    .agg(snapshots=("observed_at", "nunique"),
         first=("observed_at", "min"),
         last=("observed_at", "max"),
         price_min=("price", "min"),
         price_max=("price", "max"))
    .sort_values("snapshots", ascending=False)
)
coverage["price_range"] = coverage["price_max"] - coverage["price_min"]
print(coverage.to_string())

# %% [markdown]
# ## Thesis A — do prices move?
# Plot each itinerary's cheapest fare over calendar time. If the lines are flat,
# there's nothing to predict and the premise needs rethinking. We also summarize
# movement numerically (range and % swing) so it isn't purely eyeballed.

# %%
movement = coverage.copy()
movement["pct_swing"] = (movement["price_range"] / movement["price_min"] * 100).round(1)
print("Price movement per itinerary (sorted by % swing):")
print(movement[["snapshots", "price_min", "price_max", "price_range", "pct_swing"]]
      .sort_values("pct_swing", ascending=False).to_string())

# %%
# One line per itinerary, faceted loosely by route for readability.
for dest in sorted(mins.dest.unique()):
    sub = mins[mins.dest == dest]
    fig, ax = plt.subplots(figsize=(10, 4))
    for itin, g in sub.groupby("itinerary"):
        if g.observed_at.nunique() < 2:
            continue  # can't draw a trajectory from a single point
        label = itin.split("|", 1)[1]  # drop the redundant dest prefix
        ax.plot(g.observed_at, g.price, marker="o", ms=3, label=label)
    ax.set_title(f"{dest}: cheapest fare over time, by carrier x date-pair")
    ax.set_xlabel("observed_at (UTC)")
    ax.set_ylabel("min round-trip price ($)")
    ax.legend(fontsize=7)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Thesis B — Delta vs Frontier co-movement
# Your signature feature is "the competitor's price predicts this carrier's price."
# For each route x date-pair where both carriers appear, plot their cheapest fares
# together and correlate their *changes* (diffs, not levels — levels can correlate
# just from a shared trend). Positive diff-correlation = the carriers react together.

# %%
pairs = sorted(set(zip(mins.dest, mins.dep_date.dt.strftime("%Y-%m-%d"),
                       mins.ret_date.dt.strftime("%Y-%m-%d"))))
for dest, dep, ret in pairs:
    piv = T.pivot_min_by_carrier(mins, dest, dep, ret)
    if piv.shape[1] < 2 or len(piv) < 3:
        continue  # need both carriers and enough points to say anything
    fig, ax = plt.subplots(figsize=(10, 4))
    piv.plot(ax=ax, marker="o", ms=3)
    ax.set_title(f"{dest} {dep}->{ret}: Delta vs Frontier cheapest fare")
    ax.set_ylabel("min round-trip price ($)")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

    diffs = piv.diff().dropna()
    if len(diffs) >= 2 and diffs.shape[1] == 2:
        c = diffs.corr().iloc[0, 1]
        print(f"{dest} {dep}->{ret}: corr of day-over-day changes "
              f"({diffs.columns[0]} vs {diffs.columns[1]}) = {c:+.2f}  (n={len(diffs)})")

# %% [markdown]
# ## Thesis C — upward steps in the lowest fare
# A sudden jump up in the cheapest fare is, in real time, a fare bucket selling
# out — your free substitute for ExpertFlyer's RBD counts. We diff each itinerary's
# min-price series and flag positive jumps. Frequent, sizeable steps mean the
# signal exists and is worth engineering features around.

# %%
step_rows = []
for itin, g in mins.groupby("itinerary"):
    g = g.sort_values("observed_at")
    d = g["price"].diff()
    ups = d[d > 0]
    if len(ups):
        for ts, jump in zip(g["observed_at"][d > 0], ups):
            step_rows.append({"itinerary": itin.split("|", 1)[1],
                              "dest": itin.split("|")[0],
                              "observed_at": ts, "step_up": int(jump)})
steps = pd.DataFrame(step_rows)
if len(steps):
    print(f"detected {len(steps)} upward steps across all itineraries")
    print(f"median step: ${steps.step_up.median():.0f}   max: ${steps.step_up.max():.0f}")
    print("\nlargest steps:")
    print(steps.sort_values("step_up", ascending=False).head(10).to_string(index=False))
else:
    print("no upward steps yet — expected this early; revisit as data deepens")

# %% [markdown]
# ## Bonus — time-of-day (very preliminary)
# You want to model "is the redeye usually cheapest." Far too little data to
# trust yet, but we can look at the shape: median fare by departure hour bucket.
# Uses the per-departure view (dep_hour), not the per-carrier minimum.

# %%
dep = T.departure_series(df)
dep = dep.dropna(subset=["dep_hour"])
dep["tod"] = pd.cut(
    dep["dep_hour"],
    bins=[-1, 4, 8, 11, 16, 20, 23],
    labels=["redeye (0-4)", "early (5-8)", "morning (9-11)",
            "midday (12-16)", "evening (17-20)", "late (21-23)"],
)
tod = dep.groupby(["carrier", "tod"], observed=True)["price"].median().unstack("carrier")
print("Median fare by departure time-of-day (PRELIMINARY — thin data):")
print(tod.round(0).to_string())

# %% [markdown]
# ## Read the results
# - **Thesis A holds** if pct_swing is non-trivial (say >5-10%) on most itineraries.
#   Flat lines everywhere => reconsider route/date-pair selection.
# - **Thesis B holds** if diff-correlations are mostly positive. Noisy at n<10;
#   the sign mattering more than the magnitude this early.
# - **Thesis C holds** if upward steps appear at all and are non-tiny.
# - **Time-of-day** is a shape preview only; do not draw conclusions yet.
#
# Whatever the verdict, the value is catching a broken assumption now rather than
# after building a model on it. Re-run this notebook weekly as the panel deepens.
