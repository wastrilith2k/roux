"""
Weather Tools -- current conditions and forecast via Open-Meteo (free, no key).

WHAT: get_current_weather() and get_forecast() for any location by name.
      Returns temperature (F), feels-like, humidity, wind, conditions, and
      for forecasts: daily high/low, precipitation chance, sunrise/sunset.

WHY:  Weather is a natural conversation topic and useful context for the
      companion ("it's raining there, maybe suggest indoor activities").
      Open-Meteo requires no API key, making it zero-friction to deploy.

HOW:  Location name -> geocode via Open-Meteo geocoder -> weather API call
      with lat/lon. WMO weather codes are mapped to human-readable strings.
      Default location is the user's home area (Sunnyside, OR).
"""

import requests
from typing import Dict, List, Optional


# WMO Weather interpretation codes
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _geocode(location: str) -> Dict:
    """Convert a location name to coordinates using Open-Meteo geocoding."""
    # Open-Meteo geocoder doesn't handle "City, ST" well — try as-is first,
    # then fall back to just the city part
    for query in [location, location.split(",")[0].strip()]:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            break
    if not results:
        raise ValueError(f"Location not found: {location}")
    r = results[0]
    return {
        "name": r.get("name", location),
        "region": r.get("admin1", ""),
        "country": r.get("country", ""),
        "latitude": r["latitude"],
        "longitude": r["longitude"],
        "timezone": r.get("timezone", "auto"),
    }


def get_current_weather(location: str = "Sunnyside, OR") -> Dict:
    """Get current weather for a location.

    Args:
        location: City/place name (default: Sunnyside, OR)

    Returns:
        Dict with temperature, feels_like, humidity, wind_speed,
        conditions, location info

    Example:
        >>> from tools import weather
        >>> weather.get_current_weather("Portland, OR")
        >>> weather.get_current_weather()  # default location
    """
    try:
        geo = _geocode(location)
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": geo["timezone"],
            },
            timeout=10,
        )
        resp.raise_for_status()
        current = resp.json().get("current", {})

        code = current.get("weather_code", 0)
        return {
            "location": f"{geo['name']}, {geo['region']}",
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "wind_speed": current.get("wind_speed_10m"),
            "conditions": _WMO_CODES.get(code, f"Code {code}"),
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Weather fetch failed: {e}"}


def get_forecast(location: str = "Sunnyside, OR", days: int = 3) -> List[Dict]:
    """Get daily weather forecast for a location.

    Args:
        location: City/place name (default: Sunnyside, OR)
        days: Number of forecast days (1-7, default 3)

    Returns:
        List of daily forecasts with date, high, low, conditions,
        precipitation_chance, sunrise, sunset

    Example:
        >>> from tools import weather
        >>> weather.get_forecast("San Francisco", days=5)
    """
    days = max(1, min(days, 7))
    try:
        geo = _geocode(location)
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
                "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max,sunrise,sunset",
                "temperature_unit": "fahrenheit",
                "timezone": geo["timezone"],
                "forecast_days": days,
            },
            timeout=10,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily", {})

        dates = daily.get("time", [])
        forecast = []
        for i, date in enumerate(dates):
            code = daily["weather_code"][i] if i < len(daily.get("weather_code", [])) else 0
            forecast.append({
                "date": date,
                "high": daily.get("temperature_2m_max", [None])[i],
                "low": daily.get("temperature_2m_min", [None])[i],
                "conditions": _WMO_CODES.get(code, f"Code {code}"),
                "precipitation_chance": daily.get("precipitation_probability_max", [None])[i],
                "sunrise": daily.get("sunrise", [""])[i].split("T")[-1] if daily.get("sunrise") else "",
                "sunset": daily.get("sunset", [""])[i].split("T")[-1] if daily.get("sunset") else "",
            })

        return forecast
    except ValueError as e:
        return [{"error": str(e)}]
    except Exception as e:
        return [{"error": f"Forecast fetch failed: {e}"}]
