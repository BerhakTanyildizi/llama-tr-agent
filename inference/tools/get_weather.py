# -*- coding: utf-8 -*-
"""Weather tool. Provider: Open-Meteo (no API key required).

Swapping providers touches _fetch() only; the run() contract stays.

SCHEMA DEVIATES FROM THE TRAINING DATA, deliberately: `days_ahead` was added so
the user can ask about tomorrow or "in 3 days". Everywhere else in this project
a schema that differs from training is treated as a bug, so the reasoning:

  * The eval measured the model on 19 records built from tool schemas it had
    NEVER seen and it scored 94% - equal to its score on familiar ones. Reading
    a new parameter out of a schema is precisely the ability that was measured
    and confirmed, so this extension is low risk.
  * days_ahead is an INTEGER OFFSET, not a date string. The model was measured
    fabricating dates (eval E04 produced date='2022-03-15' while the real date
    sat in its own system prompt). "2 gün sonra" -> 2 is read straight off the
    user's words and can be grounded; a date has to be computed and was not.
  * eval/test_set.jsonl carries the same extended schema, and the validator has
    an explicit allowlist entry, so accidental drift is still caught.

Raw provider JSON is never put into the context: only the fields the model
needs to form a sentence are returned (context budget, CLAUDE.md item 4).
"""
import json, unicodedata, urllib.parse, urllib.request

SCHEMA = {"type": "function", "function": {"name": "get_weather", "description": "Get current or forecast weather conditions for a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city name, e.g. 'Ankara'."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit."}, "days_ahead": {"type": "integer", "description": "How many days from today, 0 for today's current conditions, 1 for tomorrow, up to 7. Omit for current conditions."}}, "required": ["location"]}}}

# 'location' is EXTRACTED from what the user said, never composed. If the model
# produces a city the user never mentioned it is a fabrication (eval E01).
GROUNDED_PARAMS = frozenset({"location"})

TIMEOUT = 10
MAX_DAYS_AHEAD = 7       # provider serves more, but forecast quality falls off
# WMO weather code -> short English description. The model translates it into
# Turkish itself (hybrid language strategy: tool output English, answer Turkish).
WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "rain", 65: "heavy rain",
    71: "slight snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}


def _fetch(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return json.loads(r.read())


# --- location resolution ---------------------------------------------------
# The model writes city names without Turkish diacritics ("Elazig"), and the
# geocoder returns nothing for some of them - observed live, the agent asked
# the user to repeat the city three times in a row. Two fixes, both measured:
#
#   1. If the plain query yields no name that folds to the query, retry with a
#      shortened query and keep only candidates whose folded name matches. Blind
#      truncation alone is unsafe ("Mugla" -> "Mulegé"); the fold check rejects
#      those. "Xyzabad" still resolves to nothing, so the error path survives.
#   2. Prefer the most populous match. Without it "Diyarbakir" resolved to the
#      district "Diyarbakir Northwest" instead of the city.
_TR_FOLD = str.maketrans("ıİğĞşŞçÇöÖüÜâÂîÎûÛ", "iigGsScCoOuUaAiIuU")


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.translate(_TR_FOLD))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def _geocode(name: str) -> list[dict]:
    q = urllib.parse.urlencode({"name": name, "count": 10, "language": "tr"})
    return _fetch(f"https://geocoding-api.open-meteo.com/v1/search?{q}").get("results") or []


def _resolve(location: str) -> dict | None:
    direct = _geocode(location)
    target = _fold(location)
    exact = [c for c in direct if _fold(c["name"]) == target]
    if not exact and len(location) >= 5:
        exact = [c for c in _geocode(location[:-2]) if _fold(c["name"]) == target]
    pool = exact or direct
    return max(pool, key=lambda c: c.get("population") or 0) if pool else None


def run(location: str, unit: str = "celsius", days_ahead: int = 0) -> dict:
    """days_ahead=0 -> current conditions; 1..MAX_DAYS_AHEAD -> daily forecast.

    The two branches return DIFFERENT shapes on purpose. A forecast has no
    "current temperature": reporting a daily max as if it were the temperature
    right now would be a quiet lie, so a forecast returns min/max plus a
    precipitation probability and carries an explicit `date`. The model can
    then say "yarın 26 dereceye kadar çıkacak" instead of "yarın 26 derece".
    """
    temp_unit = "fahrenheit" if str(unit).lower().startswith("f") else "celsius"

    try:
        day = int(days_ahead or 0)
    except (TypeError, ValueError):
        return {"error": "invalid_days_ahead",
                "message": f"days_ahead must be a whole number, got {days_ahead!r}."}
    if not 0 <= day <= MAX_DAYS_AHEAD:
        # Refused rather than clamped: silently answering about a different day
        # than the user asked about is worse than saying it cannot be done.
        return {"error": "day_out_of_range",
                "message": (f"Forecasts are available for today up to {MAX_DAYS_AHEAD} "
                            f"days ahead; {day} was requested.")}

    try:
        place = _resolve(location)
        if place is None:
            return {"error": "location_not_found",
                    "message": f"Could not resolve location: '{location}'."}

        params = {"latitude": place["latitude"], "longitude": place["longitude"],
                  "temperature_unit": temp_unit, "timezone": "auto"}
        if day == 0:
            params["current"] = "temperature_2m,relative_humidity_2m,weather_code"
        else:
            params["daily"] = ("temperature_2m_max,temperature_2m_min,weather_code,"
                               "precipitation_probability_max")
            params["forecast_days"] = day + 1
        data = _fetch("https://api.open-meteo.com/v1/forecast?"
                      + urllib.parse.urlencode(params))
    except Exception as e:
        return {"error": "provider_unavailable", "message": str(e)[:120]}

    common = {"location": place["name"], "country": place.get("country", ""),
             "unit": temp_unit}

    if day == 0:
        cur = data["current"]
        return {**common, "when": "now",
                "temperature": cur["temperature_2m"],
                "humidity_percent": cur["relative_humidity_2m"],
                "condition": WMO.get(cur["weather_code"], "unknown")}

    d = data["daily"]
    if day >= len(d["time"]):
        return {"error": "day_unavailable",
                "message": f"The provider returned no forecast for day +{day}."}
    return {**common, "when": f"+{day} day(s)", "date": d["time"][day],
            "days_ahead": day,
            "temperature_min": d["temperature_2m_min"][day],
            "temperature_max": d["temperature_2m_max"][day],
            "precipitation_probability_percent": d["precipitation_probability_max"][day],
            "condition": WMO.get(d["weather_code"][day], "unknown")}
