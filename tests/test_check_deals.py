"""Mistake-fare detector tests on synthetic panels — no DB, no network.

find_deals() judges only the LATEST snapshot of each itinerary against its
trailing-min baseline, so each fixture below sets up a history then a final
"live" price and asserts whether it should fire.
"""

import pandas as pd
import pytest

import check_deals as D
import trajectories as T


def _raw(observed_at, dest, carrier, price, dep="2026-07-10", ret="2026-07-13"):
    return {
        "observed_at": observed_at, "origin": "ATL", "dest": dest,
        "dep_date": dep, "ret_date": ret, "carrier": carrier, "cabin": "economy",
        "price": price, "stops": 0, "nonstop": True, "duration_min": 101,
        "dep_hour": 8, "dep_minute": 0, "dep_dow": "Fri", "price_level": None,
    }


def _mins(rows):
    return T.carrier_min_series(T._coerce_types(pd.DataFrame(rows)))


def test_clear_mistake_fare_fires():
    # Trailing min ~200, then a live $150 = 25% / $50 drop -> a deal.
    rows = [
        _raw("2026-06-25T10:00:00+00:00", "MCO", "Delta", 200),
        _raw("2026-06-26T10:00:00+00:00", "MCO", "Delta", 210),
        _raw("2026-06-27T10:00:00+00:00", "MCO", "Delta", 205),
        _raw("2026-06-28T10:00:00+00:00", "MCO", "Delta", 150),
    ]
    deals = D.find_deals(_mins(rows))
    assert len(deals) == 1
    d = deals.iloc[0]
    assert d["dest"] == "MCO"
    assert d["latest_price"] == 150
    assert d["trailing_min"] == 200
    assert d["drop_dollars"] == 50
    assert d["drop_pct"] == pytest.approx(0.25)


def test_ordinary_wiggle_does_not_fire():
    # Live price barely under the trailing min -> below threshold, stay quiet.
    rows = [
        _raw("2026-06-25T10:00:00+00:00", "LAS", "Frontier", 200),
        _raw("2026-06-26T10:00:00+00:00", "LAS", "Frontier", 195),
        _raw("2026-06-27T10:00:00+00:00", "LAS", "Frontier", 198),
        _raw("2026-06-28T10:00:00+00:00", "LAS", "Frontier", 190),  # only ~2.5% off
    ]
    assert D.find_deals(_mins(rows)).empty


def test_insufficient_history_stays_silent():
    # Big drop, but fewer than MIN_PRIOR prior snapshots -> no trustworthy baseline.
    rows = [
        _raw("2026-06-27T10:00:00+00:00", "DEN", "Delta", 300),
        _raw("2026-06-28T10:00:00+00:00", "DEN", "Delta", 150),
    ]
    assert D.find_deals(_mins(rows)).empty


def test_small_dollar_drop_is_ignored():
    # 30% off but only a $15 drop on a cheap fare -> below MIN_DROP_DOLLARS.
    rows = [
        _raw("2026-06-25T10:00:00+00:00", "MCO", "Frontier", 50),
        _raw("2026-06-26T10:00:00+00:00", "MCO", "Frontier", 52),
        _raw("2026-06-27T10:00:00+00:00", "MCO", "Frontier", 51),
        _raw("2026-06-28T10:00:00+00:00", "MCO", "Frontier", 35),  # 30% but only $15
    ]
    assert D.find_deals(_mins(rows)).empty


def test_baseline_ignores_prices_outside_the_window():
    # A cheap $100 from 20 days ago is OUTSIDE the 14d window, so the baseline is
    # the recent ~200 band and the live $150 still reads as a real deal.
    rows = [
        _raw("2026-06-08T10:00:00+00:00", "MCO", "Delta", 100),  # 20 days before latest
        _raw("2026-06-25T10:00:00+00:00", "MCO", "Delta", 200),
        _raw("2026-06-26T10:00:00+00:00", "MCO", "Delta", 210),
        _raw("2026-06-27T10:00:00+00:00", "MCO", "Delta", 205),
        _raw("2026-06-28T10:00:00+00:00", "MCO", "Delta", 150),
    ]
    deals = D.find_deals(_mins(rows))
    assert len(deals) == 1
    assert deals.iloc[0]["trailing_min"] == 200  # the old $100 did not lower it


def test_empty_frame_is_safe():
    empty = pd.DataFrame(
        columns=["itinerary", "observed_at", "price", "dest", "carrier", "dep_date"]
    )
    assert D.find_deals(empty).empty
