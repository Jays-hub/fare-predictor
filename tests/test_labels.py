"""Label-builder tests on hand-computable panels — no DB, no network.

Censoring is the whole point of labels.py, so most tests pin down WHEN a label
is allowed to exist, not just what its value is. The workhorse series is the
familiar 200 -> 180 -> 190 dip-and-rebound over three daily snapshots.
"""

import numpy as np
import pandas as pd
import pytest

import labels as L
import trajectories as T

S1 = "2026-06-05T10:00:00+00:00"
S2 = "2026-06-06T10:00:00+00:00"
S3 = "2026-06-07T10:00:00+00:00"


def _raw(observed_at, price, dest="MCO", dep="2026-06-26", ret="2026-06-29",
         carrier="Frontier"):
    return {
        "observed_at": observed_at, "origin": "ATL", "dest": dest,
        "dep_date": dep, "ret_date": ret, "carrier": carrier, "cabin": "economy",
        "price": price, "stops": 0, "nonstop": True, "duration_min": 101,
        "dep_hour": 8, "dep_minute": 0, "dep_dow": "Fri", "price_level": None,
    }


def _mins(rows):
    return T.carrier_min_series(T._coerce_types(pd.DataFrame(rows)))


def _col(out, carrier, col):
    sub = out[out.carrier == carrier].sort_values("observed_at")
    return sub[col].tolist()


@pytest.fixture
def dip_panel():
    return _mins([_raw(S1, 200), _raw(S2, 180), _raw(S3, 190)])


def test_grain_and_column_contract(dip_panel):
    out = L.build_labels(dip_panel)
    assert len(out) == len(dip_panel)                       # one row in, one out
    assert set(out.columns) - set(dip_panel.columns) == set(L.LABEL_COLUMNS)


def test_forward_window_values(dip_panel):
    # H=1 day: each row's window holds exactly the next daily snapshot.
    out = L.build_labels(dip_panel, drop_dollars=20, horizon_days=1)
    fwd = _col(out, "Frontier", "fwd_min")
    assert fwd[:2] == [180, 190] and np.isnan(fwd[2])       # last row: no future
    assert _col(out, "Frontier", "fwd_n_snapshots") == [1, 1, 0]
    assert _col(out, "Frontier", "label_observed") == [True, True, False]
    wd = _col(out, "Frontier", "will_drop")
    # S1: 180 <= 200-20 — a drop of EXACTLY $20 fires. S2: 190 vs 160 doesn't.
    assert wd[0] == 1 and wd[1] == 0 and pd.isna(wd[2])


def test_right_edge_is_censored_even_with_a_visible_drop(dip_panel):
    # Default H=7d on a 2-day panel: no window has elapsed, nothing departed.
    # S1 already SEES the $20 drop, but keeping early-confirmed 1s while the
    # undetermined 0s stay excluded would over-sample positives at the edge —
    # so every label is NA until its window closes.
    out = L.build_labels(dip_panel)
    assert not out["label_observed"].any()
    assert out["will_drop"].isna().all()


def test_departed_itinerary_unlocks_windows():
    # A (Frontier) departs 06-08 — BEFORE the panel ends (B keeps collecting
    # to 06-10) — so A's windows can never change: observed despite H=7d.
    rows = [
        _raw(S1, 200, dep="2026-06-08", ret="2026-06-11"),
        _raw(S2, 180, dep="2026-06-08", ret="2026-06-11"),
        _raw(S3, 190, dep="2026-06-08", ret="2026-06-11"),
    ] + [
        _raw(f"2026-06-{d:02d}T10:00:00+00:00", 300, dest="LAS",
             dep="2026-07-31", ret="2026-08-03", carrier="Delta")
        for d in range(5, 11)
    ]
    out = L.build_labels(_mins(rows), drop_dollars=20, horizon_days=7)

    assert _col(out, "Frontier", "label_observed") == [True, True, True]
    wd = _col(out, "Frontier", "will_drop")
    assert wd[0] == 1          # min(180, 190) <= 200 - 20
    assert wd[1] == 0          # 190 vs 180-20: rebound, no qualifying drop
    # Deadline row: empty window is a GENUINE 0, not censoring — the itinerary
    # departed and no lower fare ever materialized.
    assert wd[2] == 0
    assert _col(out, "Frontier", "fwd_n_snapshots")[2] == 0

    # B is live and its 7d windows haven't elapsed: fully censored.
    assert pd.Series(_col(out, "Delta", "will_drop")).isna().all()


def test_departure_day_itself_is_not_departed():
    # C departs ON the panel-end date (06-10): a cycle may still run that day,
    # so C does NOT count as departed and its unelapsed windows stay censored.
    rows = [
        _raw("2026-06-08T10:00:00+00:00", 200, dep="2026-06-10", ret="2026-06-13"),
        _raw("2026-06-09T10:00:00+00:00", 150, dep="2026-06-10", ret="2026-06-13"),
        _raw("2026-06-10T10:00:00+00:00", 300, dest="LAS", dep="2026-07-31",
             ret="2026-08-03", carrier="Delta"),
    ]
    out = L.build_labels(_mins(rows), horizon_days=7)
    assert _col(out, "Frontier", "label_observed") == [False, False]
    assert pd.Series(_col(out, "Frontier", "will_drop")).isna().all()


def test_window_excludes_rows_beyond_horizon():
    # Snapshots 3 days apart with H=2: the later row exists but falls OUTSIDE
    # the window — and the elapsed empty window is a genuine 0.
    rows = [
        _raw("2026-06-01T10:00:00+00:00", 200),
        _raw("2026-06-04T10:00:00+00:00", 100),
    ]
    out = L.build_labels(_mins(rows), drop_dollars=20, horizon_days=2)
    fwd = _col(out, "Frontier", "fwd_min")
    assert np.isnan(fwd[0]) and np.isnan(fwd[1])
    assert _col(out, "Frontier", "fwd_n_snapshots") == [0, 0]
    wd = _col(out, "Frontier", "will_drop")
    assert wd[0] == 0 and pd.isna(wd[1])   # row 1 elapsed-empty; row 2 unelapsed


def test_drop_threshold_param():
    rows = [_raw(S1, 200), _raw(S2, 180), _raw(S3, 190)]
    out = L.build_labels(_mins(rows), drop_dollars=25, horizon_days=1)
    # The $20 dip is no longer deep enough at delta=$25.
    assert _col(out, "Frontier", "will_drop")[0] == 0
