"""Feature-builder tests on a hand-computable two-carrier panel — no DB/network.

One competitive market (MCO 06-26/06-29) over 3 daily snapshots, so the
cross-carrier feature has a competitor to find:

  Frontier: 200 -> 180 -> 190   (a dip then partial rebound)
  Delta:    240 -> 250 -> 230

Every feature value below is checkable by hand against those two series.
"""

import numpy as np
import pandas as pd
import pytest

import features as F
import trajectories as T

S1 = "2026-06-05T10:00:00+00:00"
S2 = "2026-06-06T10:00:00+00:00"
S3 = "2026-06-07T10:00:00+00:00"


def _raw(observed_at, carrier, price, price_level="typical"):
    return {
        "observed_at": observed_at, "origin": "ATL", "dest": "MCO",
        "dep_date": "2026-06-26", "ret_date": "2026-06-29",
        "carrier": carrier, "cabin": "economy", "price": price,
        "stops": 0, "nonstop": True, "duration_min": 101,
        "dep_hour": 8, "dep_minute": 0, "dep_dow": "Fri", "price_level": price_level,
    }


@pytest.fixture
def feats():
    rows = [
        _raw(S1, "Frontier", 200), _raw(S2, "Frontier", 180), _raw(S3, "Frontier", 190),
        _raw(S1, "Delta", 240), _raw(S2, "Delta", 250), _raw(S3, "Delta", 230),
    ]
    mins = T.carrier_min_series(T._coerce_types(pd.DataFrame(rows)))
    return F.build_features(mins)


def _series(feats, carrier, col):
    sub = feats[feats.carrier == carrier].sort_values("observed_at")
    return sub[col].tolist()


def test_row_grain_is_preserved(feats):
    assert len(feats) == 6                      # one row in, one row out
    assert feats["itinerary"].nunique() == 2


def test_lag_delta_and_trailing_min(feats):
    lag = _series(feats, "Frontier", "price_lag1")
    assert np.isnan(lag[0])                      # no prior snapshot at the first row
    assert lag[1:] == [200, 180]
    assert _series(feats, "Frontier", "price_delta1")[1:] == [-20, 10]
    assert _series(feats, "Frontier", "trailing_min") == [200, 180, 180]
    assert _series(feats, "Frontier", "vs_trailing_min") == [0, 0, 10]
    assert _series(feats, "Frontier", "snapshots_seen") == [1, 2, 3]


def test_rolling_min(feats):
    # min over the trailing window (incl. current), min_periods=1.
    assert _series(feats, "Frontier", "roll_min_3") == [200, 180, 180]
    assert _series(feats, "Delta", "roll_min_3") == [240, 240, 230]


def test_up_steps(feats):
    # Frontier moves down then up: one up-step on the last snapshot.
    assert _series(feats, "Frontier", "is_up_step") == [0, 0, 1]
    assert _series(feats, "Frontier", "n_up_steps") == [0, 0, 1]
    assert _series(feats, "Frontier", "last_up_step")[2] == 10
    # Delta moves up then down: up-step on the middle snapshot.
    assert _series(feats, "Delta", "n_up_steps") == [0, 1, 1]


def test_days_min_held(feats):
    # Frontier: new min S1, new min S2, then held -> 1,1,2.
    assert _series(feats, "Frontier", "days_min_held") == [1, 1, 2]
    # Delta: new min S1, held S2, new min S3 -> 1,2,1.
    assert _series(feats, "Delta", "days_min_held") == [1, 2, 1]


def test_cross_carrier_competitor(feats):
    # Each carrier sees the OTHER's same-snapshot price.
    assert _series(feats, "Frontier", "competitor_price") == [240, 250, 230]
    assert _series(feats, "Delta", "competitor_price") == [200, 180, 190]
    assert _series(feats, "Frontier", "price_vs_competitor") == [-40, -70, -40]
    assert _series(feats, "Delta", "price_vs_competitor") == [40, 70, 40]


def test_competitor_is_nan_when_other_carrier_absent():
    # Only one carrier in the market -> no competitor to find.
    rows = [_raw(S1, "Frontier", 200), _raw(S2, "Frontier", 180)]
    mins = T.carrier_min_series(T._coerce_types(pd.DataFrame(rows)))
    feats = F.build_features(mins)
    assert feats["competitor_price"].isna().all()


def test_features_are_causal_no_lookahead(feats):
    # The trailing min at each snapshot must never undercut a price that only
    # appears later — i.e. it equals the min of THIS row and earlier ones only.
    for carrier in ("Frontier", "Delta"):
        prices = _series(feats, carrier, "price")
        tmin = _series(feats, carrier, "trailing_min")
        assert tmin == [min(prices[: i + 1]) for i in range(len(prices))]
