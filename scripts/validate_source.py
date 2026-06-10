from fast_flights import FlightData, Passengers, Result, get_flights

result = get_flights(
    flight_data=[
        FlightData(
            date="2026-06-18",   # Designed Date
            from_airport="ATL",  # Atlanta
            to_airport="MCO",    # Orlando
        ),
        FlightData(
            date="2026-06-25",   # Designed Date
            from_airport="MCO",  # Orlando
            to_airport="ATL",    # Atlanta
        ),
    ],
    seat="economy",  # business/economy/first/premium-economy
    trip="round-trip",  # multi-city/one-way/round-trip
    passengers=Passengers(adults=1),
    #fetch_mode="fallback",  # fast/fallback
)

print(result)