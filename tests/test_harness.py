"""Harness tests on a hand-computable synthetic panel — no DB, no network.

Two itineraries with prices chosen so every dollar figure is checkable by hand:

  A) MCO/Frontier, dep 06-26: prices 200 -> 180 -> 190 over 3 daily snapshots
     (days_to_dep 21, 20, 19). A real dip then partial rebound.
  B) LAS/Delta,    dep 07-10: prices 300 -> 320 -> 330 (days 14, 13, 12).
     Monotonic rise: buy-now is already optimal; waiting only loses money.
"""

import pandas as pd
import pytest

import harness as H
import trajectories as T


def _raw(observed_at, dest, dep, ret, carrier, price):
    return {
        "observed_at": observed_at, "origin": "ATL", "dest": dest,
        "dep_date": dep, "ret_date": ret, "carrier": carrier, "cabin": "economy",
        "price": price, "stops": 0, "nonstop": True, "duration_min": 101,
        "dep_hour": 8, "dep_minute": 0, "dep_dow": "Fri", "price_level": None,
    }


@pytest.fixture
def mins():
    rows = [
        _raw("2026-06-05T10:00:00+00:00", "MCO", "2026-06-26", "2026-06-29", "Frontier", 200),
        _raw("2026-06-06T10:00:00+00:00", "MCO", "2026-06-26", "2026-06-29", "Frontier", 180),
        _raw("2026-06-07T10:00:00+00:00", "MCO", "2026-06-26", "2026-06-29", "Frontier", 190),
        _raw("2026-06-26T10:00:00+00:00", "LAS", "2026-07-10", "2026-07-13", "Delta", 300),
        _raw("2026-06-27T10:00:00+00:00", "LAS", "2026-07-10", "2026-07-13", "Delta", 320),
        _raw("2026-06-28T10:00:00+00:00", "LAS", "2026-07-10", "2026-07-13", "Delta", 330),
    ]
    df = T._coerce_types(pd.DataFrame(rows))
    return T.carrier_min_series(df)


def _by_itin(outcomes):
    return outcomes.set_index("dest").to_dict("index")


def test_always_buy_now(mins):
    out = _by_itin(H.simulate(mins, H.always_buy_now))
    # Buys at first snapshot of each itinerary -> saves nothing.
    assert out["MCO"]["policy_price"] == 200
    assert out["MCO"]["saved_vs_buynow"] == 0
    assert out["MCO"]["oracle_savings"] == 20   # could have caught the 180 dip
    assert out["LAS"]["policy_price"] == 300
    assert out["LAS"]["oracle_savings"] == 0    # rising fare, no room to save

    s = H.summarize(H.simulate(mins, H.always_buy_now))
    assert s["total_saved_vs_buynow"] == 0
    assert s["total_oracle_savings"] == 20
    assert s["pct_oracle_captured"] == 0.0      # captures 0% of oracle by def'n


def test_deadline_buy_can_lose_money(mins):
    out = _by_itin(H.simulate(mins, H.never_buy))
    # Forced to buy at the last snapshot of each itinerary.
    assert out["MCO"]["policy_price"] == 190
    assert out["MCO"]["saved_vs_buynow"] == 10
    assert out["LAS"]["policy_price"] == 330
    assert out["LAS"]["saved_vs_buynow"] == -30   # waiting LOST $30 on the riser

    s = H.summarize(H.simulate(mins, H.never_buy))
    assert s["total_saved_vs_buynow"] == -20
    assert s["pct_oracle_captured"] == -1.0       # -20 saved / 20 of headroom
    assert s["n_lost_to_buynow"] == 1


def test_below_trailing_median_catches_the_dip(mins):
    policy = H.buy_below_trailing_median(1.0, min_history=2)
    out = _by_itin(H.simulate(mins, policy))
    # MCO: snap1 waits (no history); snap2 price 180 <= median(prior=[200])=200
    # -> buys the dip exactly at the oracle price.
    assert out["MCO"]["policy_price"] == 180
    assert out["MCO"]["saved_vs_buynow"] == 20
    assert out["MCO"]["buy_days_to_dep"] == 20
    assert out["MCO"]["pct_oracle"] == 1.0        # captured 100% of MCO oracle
    # LAS: every fare is above the trailing median -> never fires -> deadline 330.
    assert out["LAS"]["policy_price"] == 330
    assert out["LAS"]["saved_vs_buynow"] == -30

    s = H.summarize(H.simulate(mins, policy))
    assert s["total_saved_vs_buynow"] == -10      # +20 (MCO) - 30 (LAS)
    assert s["pct_oracle_captured"] == -0.5       # -10 / 20


def test_fixed_days_out(mins):
    out = _by_itin(H.simulate(mins, H.fixed_days_out(19)))
    # MCO days 21,20,19 -> fires at 19 (price 190). LAS days 14.. <=19 -> first (300).
    assert out["MCO"]["policy_price"] == 190
    assert out["MCO"]["buy_days_to_dep"] == 19
    assert out["LAS"]["policy_price"] == 300
    assert out["LAS"]["buy_days_to_dep"] == 14


def test_policy_only_ever_sees_causal_history(mins):
    seen = []

    def spy(history):
        # Each call's history must be a time-sorted prefix ending at "now".
        assert history["observed_at"].is_monotonic_increasing
        seen.append((history["itinerary"].iloc[-1], len(history)))
        return False  # never buy, so we visit every snapshot

    H.simulate(mins, spy)
    # Visited each itinerary's snapshots as growing prefixes 1,2,3.
    mco = [n for itin, n in seen if itin.startswith("MCO")]
    assert mco == [1, 2, 3]


def test_split_by_dep_date_keeps_itineraries_whole(mins):
    train, test = H.split_by_dep_date(mins, "2026-07-01")
    assert set(train["dest"]) == {"MCO"}          # dep 06-26 < cutoff
    assert set(test["dest"]) == {"LAS"}           # dep 07-10 >= cutoff
    # No itinerary appears on both sides.
    assert set(train["itinerary"]) & set(test["itinerary"]) == set()
