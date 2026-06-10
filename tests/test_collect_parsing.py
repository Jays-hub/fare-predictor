"""Pure-function tests for the collector's parsing layer.

No network, no DB: parse_result is fed fake fast-flights objects (anything
with .name/.departure/.arrival/.duration/.stops/.price attributes works).
"""

from types import SimpleNamespace

import collect


# ------------------------------------------------------------- field parsers

def test_parse_price():
    assert collect._parse_price("$1,234") == 1234
    assert collect._parse_price("$89") == 89
    assert collect._parse_price("$0") is None          # zero = no real fare
    assert collect._parse_price("") is None
    assert collect._parse_price("Price unavailable") is None


def test_duration_to_minutes():
    assert collect._duration_to_minutes("1 hr 40 min") == 100
    assert collect._duration_to_minutes("2 hr") == 120
    assert collect._duration_to_minutes("55 min") == 55
    assert collect._duration_to_minutes("") is None
    assert collect._duration_to_minutes("nonsense") is None


def test_parse_departure():
    out = collect._parse_departure("5:05 AM on Thu, Jun 18")
    assert out == {"dep_time_raw": "5:05 AM on Thu, Jun 18",
                   "dep_hour": 5, "dep_minute": 5, "dep_dow": "Thu"}
    assert collect._parse_departure("12:34 PM on Fri, Jun 26")["dep_hour"] == 12
    assert collect._parse_departure("12:05 AM on Mon, Jul 20")["dep_hour"] == 0
    empty = collect._parse_departure("")
    assert empty["dep_hour"] is None and empty["dep_dow"] is None


def test_live_date_pairs_drops_expired_keeps_today_and_future():
    pairs = [("2026-06-01", "2026-06-04"),   # departed
             ("2026-06-10", "2026-06-13"),   # departs today — still observable
             ("2026-07-20", "2026-07-27")]   # future
    assert collect.live_date_pairs(pairs, "2026-06-10") == pairs[1:]


# --------------------------------------------------------------- parse_result

def _flight(name="Frontier", departure="5:05 AM on Fri, Jun 26",
            arrival="6:46 AM on Fri, Jun 26", duration="1 hr 41 min",
            stops=0, price="$120"):
    return SimpleNamespace(name=name, departure=departure, arrival=arrival,
                           duration=duration, stops=stops, price=price)


def _result(flights, current_price="low"):
    return SimpleNamespace(current_price=current_price, flights=flights)


def _parse(result, **overrides):
    kwargs = dict(origin="ATL", dest="MCO",
                  dep_date="2026-06-26", ret_date="2026-06-29",
                  observed_at="2026-06-10T10:00:00+00:00")
    kwargs.update(overrides)
    return collect.parse_result(result, **kwargs)


def test_parse_result_row_schema_matches_fieldnames():
    rows = _parse(_result([_flight()]))
    assert len(rows) == 1
    assert set(rows[0]) == set(collect.FIELDNAMES)


def test_parse_result_dedups_best_listed_twice():
    f = _flight()
    rows = _parse(_result([f, _flight()]))   # identical attrs = same flight
    assert len(rows) == 1


def test_parse_result_filters_to_tracked_carriers():
    rows = _parse(_result([_flight(name="Southwest"), _flight(name=""),
                           _flight(name="Delta", price="$180")]))
    assert [r["carrier"] for r in rows] == ["Delta"]


def test_parse_result_drops_unpriceable_rows():
    assert _parse(_result([_flight(price="Price unavailable")])) == []


def test_parse_result_stamps_price_level_on_every_row():
    rows = _parse(_result([_flight(), _flight(name="Delta", price="$180")],
                          current_price="LOW"))
    assert {r["price_level"] for r in rows} == {"low"}

    rows = _parse(_result([_flight()], current_price=""))
    assert rows[0]["price_level"] is None
    rows = _parse(_result([_flight()], current_price=None))
    assert rows[0]["price_level"] is None


def test_parse_result_shares_observed_at_and_derives_nonstop():
    rows = _parse(_result([_flight(stops=0),
                           _flight(stops=1, departure="8:00 PM on Fri, Jun 26"),
                           _flight(stops="?", departure="9:00 PM on Fri, Jun 26")]))
    assert {r["observed_at"] for r in rows} == {"2026-06-10T10:00:00+00:00"}
    by_dep = {r["dep_time_raw"]: r for r in rows}
    assert by_dep["5:05 AM on Fri, Jun 26"]["nonstop"] is True
    assert by_dep["8:00 PM on Fri, Jun 26"]["nonstop"] is False
    assert by_dep["9:00 PM on Fri, Jun 26"]["stops"] is None
    assert by_dep["9:00 PM on Fri, Jun 26"]["nonstop"] is None
