"""Reshaping tests for trajectories.py on synthetic data — no DB, no network.

The synthetic panel: one route (MCO) over two snapshot cycles a day apart,
with Frontier offering two departures and Delta one. Small enough to assert
exact values, rich enough to exercise min-aggregation, days_to_dep,
flight_key stability, and the carrier pivot.
"""

import pandas as pd
import pytest

import trajectories as T

S1 = "2026-06-05T10:00:00+00:00"
S2 = "2026-06-06T10:00:00+00:00"


def _row(observed_at, carrier, price, dep_hour, dep_minute, stops=0, nonstop=True):
    return {
        "observed_at": observed_at, "origin": "ATL", "dest": "MCO",
        "dep_date": "2026-06-26", "ret_date": "2026-06-29",
        "carrier": carrier, "cabin": "economy", "price": price,
        "stops": stops, "nonstop": nonstop, "duration_min": 101,
        "dep_hour": dep_hour, "dep_minute": dep_minute, "dep_dow": "Fri",
    }


@pytest.fixture
def df():
    rows = [
        # Frontier: redeye + evening, both snapshots; min moves 120 -> 130.
        _row(S1, "Frontier", 120, 5, 5),
        _row(S1, "Frontier", 140, 20, 21, stops=1, nonstop=False),
        _row(S2, "Frontier", 130, 5, 5),
        _row(S2, "Frontier", 140, 20, 21, stops=1, nonstop=False),
        # Delta: one departure; min moves 180 -> 175.
        _row(S1, "Delta", 180, 8, 0),
        _row(S2, "Delta", 175, 8, 0),
    ]
    return T._coerce_types(pd.DataFrame(rows))


def test_coerce_types(df):
    assert str(df["observed_at"].dt.tz) == "UTC"
    assert df["dep_date"].dt.tz is None
    assert (df["dep_date"] == pd.Timestamp("2026-06-26")).all()
    assert pd.api.types.is_numeric_dtype(df["price"])


def test_itinerary_id(df):
    ids = set(T.itinerary_id(df))
    assert ids == {"MCO|2026-06-26|2026-06-29|Frontier",
                   "MCO|2026-06-26|2026-06-29|Delta"}


def test_carrier_min_series(df):
    mins = T.carrier_min_series(df)
    # 2 carriers x 2 snapshots, collapsed to the cheapest fare each.
    assert len(mins) == 4

    frontier = mins[mins.carrier == "Frontier"].sort_values("observed_at")
    assert frontier["price"].tolist() == [120, 130]
    assert frontier["n_flights"].tolist() == [2, 2]
    assert frontier["nonstop_available"].all()

    delta = mins[mins.carrier == "Delta"].sort_values("observed_at")
    assert delta["price"].tolist() == [180, 175]
    assert delta["n_flights"].tolist() == [1, 1]

    # days_to_dep on the UTC calendar date: Jun 5/6 -> Jun 26 = 21/20 days.
    assert sorted(frontier["days_to_dep"].tolist()) == [20, 21]


def test_departure_series_flight_key_is_stable_across_snapshots(df):
    dep = T.departure_series(df)
    assert len(dep) == len(df)  # per-departure view keeps every row

    key = "MCO|2026-06-26|2026-06-29|Frontier@05:05"  # zero-padded hh:mm
    redeye = dep[dep.flight_key == key].sort_values("observed_at")
    assert len(redeye) == 2     # same physical departure tracked over time
    assert redeye["price"].tolist() == [120, 130]
    assert redeye["days_to_dep"].tolist() == [21, 20]


def test_pivot_min_by_carrier(df):
    mins = T.carrier_min_series(df)
    piv = T.pivot_min_by_carrier(mins, "MCO", "2026-06-26", "2026-06-29")
    assert list(piv.columns) == ["Delta", "Frontier"]
    assert len(piv) == 2
    assert piv["Frontier"].tolist() == [120, 130]
    assert piv["Delta"].tolist() == [180, 175]

    # A pair we never collected pivots to an empty frame, not an error.
    assert T.pivot_min_by_carrier(mins, "LAS", "2026-06-26", "2026-06-29").empty
