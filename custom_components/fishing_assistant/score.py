from homeassistant.core import HomeAssistant
import datetime
from datetime import timezone as _tzutc
from zoneinfo import ZoneInfo
from typing import Dict, Optional
import aiohttp
import pandas as pd
import logging


from .fish_profiles import FISH_PROFILES
from .helpers.astro import calculate_astronomy_forecast

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_LOGGER = logging.getLogger(__name__)


def scale_score(score):
    # `score` is a weight-normalised value in [0, 1] (weights sum to 1.0).
    # A typical decent day sits around 0.70-0.80, so map the realistic
    # 0.50-0.85 band onto 0-10 instead of the old 0.50-0.90 band, which
    # combined with the best-3h-window daily score pinned almost every day at 10.
    stretched = (score - 0.5) / (0.85 - 0.5) * 10
    return max(0, min(10, round(stretched)))


def get_profile_weights(body_type: str) -> dict:
    if body_type not in ["lake", "river", "pond", "reservoir"]:
        _LOGGER.warning(f"Unknown body_type '{body_type}', defaulting to 'lake'.")
        body_type = "lake"

    weights = {
        "temp": 0.25,
        "cloud": 0.1,
        "pressure": 0.15,
        "wind": 0.1,
        "precip": 0.1,
        "twilight": 0.15,
        "solunar": 0.1,
        "moon": 0.05,
    }

    if body_type == "river":
        weights.update({
            "pressure": 0.05,
            "solunar": 0.05,
            "precip": 0.2,
        })
    elif body_type == "pond":
        weights.update({
            "temp": 0.3,
            "precip": 0.2,
            "pressure": 0.2,
        })
    elif body_type == "reservoir":
        weights.update({
            "pressure": 0.1,
            "solunar": 0.08,
            "moon": 0.07,
        })

    # Normalise so the weights always sum to 1.0. Without this the per-body
    # overrides above add to the base weights (e.g. "pond" summed to 1.2),
    # inflating the raw score and pushing it past the top of the scale.
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights


# ----------------------------
# Unit conversion helpers
# ----------------------------

def _to_celsius(value, unit):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if unit in ("°F", "F", "fahrenheit"):
        return (value - 32) * 5 / 9
    return value


def _to_hpa(value, unit):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    u = (unit or "hPa").lower()
    if u in ("mmhg", "mm hg"):
        return value * 1.33322
    if u in ("inhg", "in hg"):
        return value * 33.8639
    return value  # hPa / mbar


def _to_kmh(value, unit):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    u = (unit or "km/h").lower()
    if u in ("m/s", "ms"):
        return value * 3.6
    if u in ("mph",):
        return value * 1.60934
    if u in ("kn", "kt", "knot", "knots"):
        return value * 1.852
    return value  # km/h


def _to_mm(value, unit):
    if value is None:
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    u = (unit or "mm").lower()
    if u in ("in", "inch", "inches"):
        return value * 25.4
    return value


# ----------------------------
# Hourly data sources
# ----------------------------

async def _hourly_from_open_meteo(hass, lat, lon, tz_name, elevation) -> Optional[pd.DataFrame]:
    """Return an hourly weather DataFrame from the Open-Meteo model."""
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=6)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,pressure_msl,precipitation,windspeed_10m",
        "daily": "sunrise,sunset",
        "timezone": tz_name,
        "elevation": elevation,
        "start_date": str(today),
        "end_date": str(end_date),
    }
    # Retry once: many sensors can hit Open-Meteo at the same instant (e.g. all
    # fish upgrading together), and an occasional request is rate-limited.
    data = None
    for attempt in range(2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    OPEN_METEO_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as response:
                    if response.status != 200:
                        _LOGGER.warning("Open-Meteo API status %s (attempt %s)", response.status, attempt + 1)
                        continue
                    body = await response.json()
                    if "hourly" not in body:
                        _LOGGER.warning("Open-Meteo response missing hourly: %s", body)
                        continue
                    data = body
                    break
        except Exception as e:
            _LOGGER.warning("Open-Meteo fetch error (attempt %s): %s", attempt + 1, e)

    if data is None:
        return None

    h = data["hourly"]
    return pd.DataFrame({
        "datetime": pd.to_datetime(h["time"]),
        "temp": h["temperature_2m"],
        "cloud": h["cloudcover"],
        "pressure": h["pressure_msl"],
        "precip": h["precipitation"],
        "wind": h["windspeed_10m"],
    })


async def _hourly_from_weather_entity(hass, entity_id, tz_name) -> Optional[pd.DataFrame]:
    """Return an hourly weather DataFrame from a HA weather.* entity."""
    try:
        resp = await hass.services.async_call(
            "weather",
            "get_forecasts",
            {"entity_id": entity_id, "type": "hourly"},
            blocking=True,
            return_response=True,
        )
    except Exception as e:
        _LOGGER.warning("weather.get_forecasts failed for %s: %s", entity_id, e)
        return None

    forecast = ((resp or {}).get(entity_id) or {}).get("forecast") or []
    if not forecast:
        _LOGGER.warning("No hourly forecast returned by %s", entity_id)
        return None

    state = hass.states.get(entity_id)
    attrs = state.attributes if state else {}
    t_unit = attrs.get("temperature_unit", "°C")
    p_unit = attrs.get("pressure_unit", "hPa")
    w_unit = attrs.get("wind_speed_unit", "km/h")
    pr_unit = attrs.get("precipitation_unit", "mm")

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _tzutc.utc

    rows = []
    for item in forecast:
        dt_raw = item.get("datetime")
        if not dt_raw:
            continue
        try:
            dt = datetime.datetime.fromisoformat(dt_raw)
        except Exception:
            continue
        if dt.tzinfo is not None:
            dt = dt.astimezone(tz)
        rows.append({
            "datetime": dt.replace(tzinfo=None),
            "temp": _to_celsius(item.get("temperature"), t_unit),
            # cloud_coverage may be absent (e.g. Gismeteo) -> NaN, handled in scoring
            "cloud": item.get("cloud_coverage"),
            "pressure": _to_hpa(item.get("pressure"), p_unit),
            "precip": _to_mm(item.get("precipitation"), pr_unit),
            "wind": _to_kmh(item.get("wind_speed"), w_unit),
        })

    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


async def _backfill_today(hass, hourly, lat, lon, tz_name, elevation) -> pd.DataFrame:
    """Prepend today's already-passed hours (missing from a weather entity's
    future-only forecast) with the actual past-hours weather from Open-Meteo,
    so the morning window still shows for the current day."""
    try:
        today = datetime.date.today()
        today_rows = hourly[hourly["datetime"].dt.date == today]
        earliest = int(today_rows["datetime"].dt.hour.min()) if not today_rows.empty else 24
        if earliest <= 0:
            return hourly  # already covers the whole day

        om = await _hourly_from_open_meteo(hass, lat, lon, tz_name, elevation)
        if om is None or om.empty:
            return hourly

        fill = om[(om["datetime"].dt.date == today) & (om["datetime"].dt.hour < earliest)]
        if fill.empty:
            return hourly

        merged = pd.concat([fill, hourly], ignore_index=True)
        merged = merged.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
        return merged
    except Exception as e:
        _LOGGER.warning("Today backfill failed: %s", e)
        return hourly


async def get_fish_score_forecast(
    hass: HomeAssistant,
    fish: str,
    lat: float,
    lon: float,
    timezone: str,
    elevation: float,
    body_type: str,
    weather_source: str = "open_meteo",
) -> Dict[str, Dict[str, str | float]]:
    fish_profile = FISH_PROFILES.get(fish)
    if not fish_profile:
        _LOGGER.warning(f"No fish profile found for '{fish}'")
        return {}

    # Moon + sun event timings (already local-timezone aware)
    astro_data = await calculate_astronomy_forecast(hass, lat, lon, timezone, days=7)
    if not astro_data:
        return {}

    # Pick the hourly weather source; fall back to Open-Meteo if a chosen
    # weather entity is missing, errors, or returns too short a horizon.
    used_source = "open_meteo"
    hourly = None
    if weather_source and weather_source not in ("open_meteo", "", None):
        hourly = await _hourly_from_weather_entity(hass, weather_source, timezone)
        if hourly is None or len(hourly) < 6:
            _LOGGER.info(
                "Weather source '%s' unusable for %s, falling back to Open-Meteo",
                weather_source, fish,
            )
            hourly = None
        else:
            used_source = weather_source
    if hourly is None:
        hourly = await _hourly_from_open_meteo(hass, lat, lon, timezone, elevation)
        used_source = "open_meteo"
    if hourly is None or hourly.empty:
        return {}

    # A weather entity only forecasts future hours, so backfill today's
    # already-passed hours from Open-Meteo (which provides observed past hours).
    if used_source != "open_meteo":
        hourly = await _backfill_today(hass, hourly, lat, lon, timezone, elevation)

    hourly["date"] = hourly["datetime"].dt.date
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["pressure_trend"] = hourly["pressure"].diff()

    forecast = {}
    weights = get_profile_weights(body_type)

    for date, group in hourly.groupby("date"):
        date_str = str(date)
        scores = []
        astro = astro_data.get(date_str, {})

        for _, row in group.iterrows():
            score = _score_hour(row=row, profile=fish_profile, astro=astro, weights=weights)
            scores.append((row["hour"], score))

        if len(scores) < 3:
            continue

        # Best 3-hour rolling window (overall)
        best_avg = 0
        best_window = ("--:--", "--:--")
        for i in range(len(scores) - 2):
            avg = (scores[i][1] + scores[i+1][1] + scores[i+2][1]) / 3
            if avg > best_avg:
                best_avg = avg
                best_window = (f"{scores[i][0]:02}:00", f"{scores[i+2][0]:02}:00")

        # Daily score is the average of all hourly scores, not the best 3-hour
        # window. The best window is still reported separately as "best_window".
        day_mean = sum(s for _, s in scores) / len(scores) if scores else 0

        # Best window separately for morning (start hour < 12) and evening
        # (start hour >= 12) — fish typically bite around both dawn and dusk.
        def _best_window(pred):
            b_avg = -1.0
            b_win = None
            for i in range(len(scores) - 2):
                if not pred(scores[i][0]):
                    continue
                avg = (scores[i][1] + scores[i + 1][1] + scores[i + 2][1]) / 3
                if avg > b_avg:
                    b_avg = avg
                    b_win = f"{scores[i][0]:02}:00 – {scores[i + 2][0]:02}:00"
            return b_win

        forecast[date_str] = {
            "score": scale_score(day_mean),
            "best_window": f"{best_window[0]} – {best_window[1]}",
            "best_window_am": _best_window(lambda h: h < 12),
            "best_window_pm": _best_window(lambda h: h >= 12),
        }

    if forecast:
        forecast["meta"] = {"weather_source": used_source}

    return forecast


def _score_hour(row, profile, astro, weights: dict) -> float:
    hour = row["hour"]

    temp_score = _score_temp(row["temp"], profile["temp_range"])
    cloud = row["cloud"]
    cloud_score = 0.7 if pd.isna(cloud) else 1 - abs(cloud - profile["ideal_cloud"]) / 100
    press_score = _score_pressure_trend(row["pressure_trend"])
    wind_score = _score_wind(row["wind"])
    precip_score = _score_precip(row["precip"])

    # Astro events
    sunrise = _parse_time(astro.get("sunrise"))
    sunset = _parse_time(astro.get("sunset"))
    moon_phase = astro.get("moon_phase", 0.5)
    transit = _parse_time(astro.get("moon_transit", None))
    underfoot = _parse_time(astro.get("moon_underfoot"))
    moonrise = _parse_time(astro.get("moonrise"))
    moonset = _parse_time(astro.get("moonset"))

    twilight_score = _score_twilight(hour, sunrise, sunset)
    moon_score = _score_moon_phase(moon_phase)
    solunar_score = _score_solunar(hour, transit, underfoot, moonrise, moonset)

    return round((
        temp_score * weights["temp"] +
        cloud_score * weights["cloud"] +
        press_score * weights["pressure"] +
        wind_score * weights["wind"] +
        precip_score * weights["precip"] +
        twilight_score * weights["twilight"] +
        solunar_score * weights["solunar"] +
        moon_score * weights["moon"]
    ), 2)


# ----------------------------
# Individual scoring functions
# ----------------------------

def _score_temp(temp: float, ideal_range) -> float:
    # Note: temp is air temp in °C, used as proxy for water temp.
    if temp is None or pd.isna(temp):
        return 0.7
    low, high = ideal_range
    if temp < low:
        return max(0, (temp - (low - 10)) / 10)
    elif temp > high:
        return max(0, (high + 10 - temp) / 10)
    return 1.0

def _score_pressure_trend(trend: float) -> float:
    if pd.isna(trend):
        return 0.7
    if trend < -2:
        return 1.0
    elif trend > 2:
        return 0.4
    return 0.7

def _score_wind(speed: float) -> float:
    # Assumes km/h
    if speed is None or pd.isna(speed):
        return 0.7
    if speed < 2:
        return 0.8
    elif speed < 6:
        return 1.0
    elif speed < 10:
        return 0.6
    return 0.2

def _score_precip(amount: float) -> float:
    # Assumes mm/h
    if amount is None or pd.isna(amount):
        return 0.7
    if amount == 0:
        return 0.7
    elif amount < 1:
        return 1.0
    elif amount < 5:
        return 0.5
    return 0.2

def _score_twilight(hour: int, sunrise, sunset) -> float:
    if not sunrise or not sunset:
        return 0.7
    if abs(hour - sunrise.hour) <= 1 or abs(hour - sunset.hour) <= 1:
        return 1.0
    return 0.7

def _score_moon_phase(phase: float) -> float:
    # 0.0 = New, 0.5 = Full, 1.0 = New
    if phase is None:
        return 0.7  # Default score when moon phase data is missing
    if phase < 0.1 or phase > 0.9:
        return 1.0
    return 0.7

def _score_solunar(hour: int, transit, underfoot, moonrise, moonset) -> float:
    boost = 0
    for event in [transit, underfoot]:
        if event and abs(hour - event.hour) <= 1:
            boost += 0.5
    for event in [moonrise, moonset]:
        if event and abs(hour - event.hour) <= 1:
            boost += 0.25
    return min(1.0, 0.6 + boost)


def _parse_time(time_str: str):
    if not time_str:
        return None
    try:
        return datetime.datetime.strptime(time_str, "%H:%M").time()
    except Exception:
        return None
