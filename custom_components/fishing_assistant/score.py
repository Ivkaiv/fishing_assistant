from homeassistant.core import HomeAssistant
import datetime
from datetime import timezone as _tzutc
from zoneinfo import ZoneInfo
from typing import Dict, Optional
import aiohttp
import pandas as pd
import logging
import time


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

async def _hourly_from_open_meteo(hass, lat, lon, tz_name, elevation, model=None) -> Optional[pd.DataFrame]:
    """Return an hourly weather DataFrame from Open-Meteo for these coordinates.

    Optional `model` selects a specific weather model (e.g. ecmwf_ifs04,
    gfs_seamless, icon_seamless); None uses Open-Meteo's best_match blend."""
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=6)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,cloudcover,pressure_msl,precipitation,windspeed_10m,winddirection_10m",
        "daily": "sunrise,sunset",
        "timezone": tz_name,
        "elevation": elevation,
        "start_date": str(today),
        "end_date": str(end_date),
    }
    if model:
        params["models"] = model
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
        "wind_dir": h.get("winddirection_10m", [None] * len(h["time"])),
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
            "wind_dir": item.get("wind_bearing"),
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


def _pirateweather_credentials(hass):
    """Return (endpoint, api_key) from the user's PirateWeather integration, or None."""
    try:
        for entry in hass.config_entries.async_entries("pirateweather"):
            key = entry.data.get("api_key")
            if key:
                endpoint = (entry.data.get("endpoint") or "https://api.pirateweather.net").rstrip("/")
                return endpoint, key
    except Exception:
        pass
    return None


async def _hourly_from_pirateweather_api(hass, lat, lon, tz_name) -> Optional[pd.DataFrame]:
    """Return an hourly weather DataFrame from the PirateWeather API for these
    exact coordinates (not the home-fixed HA weather entity)."""
    creds = _pirateweather_credentials(hass)
    if not creds:
        return None
    endpoint, key = creds
    url = f"{endpoint}/forecast/{key}/{lat},{lon}"
    params = {"units": "si", "extend": "hourly", "exclude": "minutely,daily,alerts,currently"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    _LOGGER.warning("PirateWeather API status %s", resp.status)
                    return None
                data = await resp.json()
    except Exception as e:
        _LOGGER.warning("PirateWeather API error: %s", e)
        return None

    hours = (data.get("hourly") or {}).get("data") or []
    if not hours:
        return None
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _tzutc.utc
    rows = []
    for h in hours:
        ts = h.get("time")
        if ts is None:
            continue
        dt = datetime.datetime.fromtimestamp(ts, _tzutc.utc).astimezone(tz).replace(tzinfo=None)
        cloud = h.get("cloudCover")
        rows.append({
            "datetime": dt,
            "temp": h.get("temperature"),                     # si: °C
            "cloud": cloud * 100 if cloud is not None else None,  # 0-1 -> %
            "pressure": h.get("pressure"),                    # si: hPa
            "precip": h.get("precipIntensity") or 0.0,        # si: mm/h
            "wind": (h.get("windSpeed") or 0.0) * 3.6,        # si: m/s -> km/h
            "wind_dir": h.get("windBearing"),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


async def _water_temp_estimate(hass, lat, lon, tz_name, elevation) -> Optional[float]:
    """Estimate water temperature at these coordinates from the recent air
    temperature. Water has thermal inertia, so a multi-day average of air temp
    is a far better stand-in than the raw hourly air temperature."""
    today = datetime.date.today()
    start = today - datetime.timedelta(days=4)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m",
        "timezone": tz_name,
        "elevation": elevation,
        "start_date": str(start),
        "end_date": str(today),
    }
    data = None
    for attempt in range(2):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(OPEN_METEO_URL, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status != 200:
                        _LOGGER.warning("Water-temp estimate status %s (attempt %s)", resp.status, attempt + 1)
                        continue
                    data = await resp.json()
                    break
        except Exception as e:
            _LOGGER.warning("Water-temp estimate fetch error (attempt %s): %s", attempt + 1, e)
    if data is None:
        return None
    temps = [t for t in (data.get("hourly") or {}).get("temperature_2m", []) if t is not None]
    if not temps:
        return None
    return round(sum(temps) / len(temps), 1)


async def _fetch_hourly(hass, lat, lon, tz_name, elevation, weather_source):
    """Dispatch to the chosen weather backend; return (hourly_df, used_source).

    Falls back to Open-Meteo if the chosen backend errors or is too short."""
    src = weather_source
    hourly = None
    used = "open_meteo"
    if src == "pirateweather":
        hourly = await _hourly_from_pirateweather_api(hass, lat, lon, tz_name)
        if hourly is not None and len(hourly) >= 6:
            used = "pirateweather"
        else:
            hourly = None
    elif src and str(src).startswith("weather."):
        hourly = await _hourly_from_weather_entity(hass, src, tz_name)
        if hourly is not None and len(hourly) >= 6:
            used = src
        else:
            hourly = None
    elif src and str(src).startswith("open_meteo:"):
        model = str(src).split(":", 1)[1]
        hourly = await _hourly_from_open_meteo(hass, lat, lon, tz_name, elevation, model)
        if hourly is not None and not hourly.empty:
            used = src
        else:
            hourly = None
    if hourly is None:
        hourly = await _hourly_from_open_meteo(hass, lat, lon, tz_name, elevation)
        used = "open_meteo"
    return hourly, used


# Per-location cache: all fish at one water body share a single weather/astro
# fetch (avoids N identical requests and Open-Meteo rate-limiting). Keyed by the
# inputs that change the data, with a short TTL.
_LOCATION_CACHE = {}
_CACHE_TTL = 300  # seconds


async def _get_location_data(hass, lat, lon, timezone, elevation, weather_source, temp_sensor, pressure_sensor):
    """Fetch and prepare the shared per-location hourly + astro data (cached)."""
    key = (
        round(float(lat), 5), round(float(lon), 5), str(weather_source),
        str(temp_sensor), str(pressure_sensor), round(float(elevation or 0)),
    )
    now = time.monotonic()
    cached = _LOCATION_CACHE.get(key)
    if cached and cached[0] > now:
        return cached[1]

    astro_data = await calculate_astronomy_forecast(hass, lat, lon, timezone, days=7)
    if not astro_data:
        return None

    hourly, used_source = await _fetch_hourly(hass, lat, lon, timezone, elevation, weather_source)
    if hourly is None or hourly.empty:
        return None

    # Future-only sources (PirateWeather API, HA weather entities) miss today's
    # already-passed hours — backfill them from Open-Meteo's observed past hours.
    if used_source == "pirateweather" or str(used_source).startswith("weather."):
        hourly = await _backfill_today(hass, hourly, lat, lon, timezone, elevation)

    hourly["date"] = hourly["datetime"].dt.date
    hourly["hour"] = hourly["datetime"].dt.hour
    hourly["pressure_trend"] = hourly["pressure"].diff()

    hourly, local_used = await _apply_local_sensors(
        hass, hourly, temp_sensor, pressure_sensor, lat, lon, timezone, elevation
    )

    result = (hourly, astro_data, used_source, local_used)
    for k, v in list(_LOCATION_CACHE.items()):
        if v[0] <= now:
            _LOCATION_CACHE.pop(k, None)
    _LOCATION_CACHE[key] = (now + _CACHE_TTL, result)
    return result


async def get_fish_score_forecast(
    hass: HomeAssistant,
    fish: str,
    lat: float,
    lon: float,
    timezone: str,
    elevation: float,
    body_type: str,
    weather_source: str = "open_meteo",
    temp_sensor: str = None,
    pressure_sensor: str = None,
) -> Dict[str, Dict[str, str | float]]:
    fish_profile = FISH_PROFILES.get(fish)
    if not fish_profile:
        _LOGGER.warning(f"No fish profile found for '{fish}'")
        return {}

    location = await _get_location_data(
        hass, lat, lon, timezone, elevation, weather_source, temp_sensor, pressure_sensor
    )
    if location is None:
        return {}
    hourly, astro_data, used_source, local_used = location

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
        forecast["meta"] = {
            "weather_source": used_source,
            "local_temp": local_used["temp"],
            "local_pressure": local_used["pressure"],
        }

    return forecast


def _compass(deg):
    if deg is None:
        return None
    try:
        deg = float(deg)
    except (TypeError, ValueError):
        return None
    dirs = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
            "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
    return dirs[int((deg % 360) / 22.5 + 0.5) % 16]


def _moon_name(illum, waxing):
    if illum is None:
        return None
    if illum < 0.06:
        return "новолуние"
    if illum > 0.94:
        return "полнолуние"
    if 0.44 < illum < 0.56:
        return "первая четверть" if waxing else "последняя четверть"
    return ("растущая луна" if waxing else "убывающая луна")


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


async def get_current_conditions(hass, lat, lon, timezone, elevation, weather_source, temp_sensor, pressure_sensor):
    """Current weather + sun/moon/solunar at the pond's coordinates (shares the
    per-location cache with the fish score sensors)."""
    loc = await _get_location_data(hass, lat, lon, timezone, elevation, weather_source, temp_sensor, pressure_sensor)
    if loc is None:
        return {}
    hourly, astro_data, used_source, local_used = loc
    now = datetime.datetime.now()
    today = now.date()
    today_str = str(today)

    cur = hourly[(hourly["date"] == today) & (hourly["hour"] == now.hour)]
    if cur.empty:
        t = hourly[hourly["date"] == today]
        cur = t.head(1) if not t.empty else hourly.head(1)
    if cur.empty:
        return {}
    row = cur.iloc[0]

    astro = astro_data.get(today_str, {})
    tom = astro_data.get(str(today + datetime.timedelta(days=1)), {})
    illum = astro.get("moon_phase")
    illum_t = tom.get("moon_phase")
    waxing = (illum is not None and illum_t is not None and illum_t >= illum)

    pt = _num(row.get("pressure_trend"))
    trend = "стабильно" if pt is None else ("падает" if pt < -0.3 else ("растёт" if pt > 0.3 else "стабильно"))

    def r(v, nd=0):
        n = _num(v)
        return None if n is None else (round(n, nd) if nd else round(n))

    return {
        "water_temp": r(row.get("temp"), 1),
        "wind_speed": r(row.get("wind"), 1),
        "wind_dir": _compass(row.get("wind_dir")),
        "wind_dir_deg": r(row.get("wind_dir")),
        "pressure": r(row.get("pressure")),
        "pressure_trend": trend,
        "cloud": r(row.get("cloud")),
        "precip": r(row.get("precip"), 1),
        "sunrise": astro.get("sunrise"),
        "sunset": astro.get("sunset"),
        "moonrise": astro.get("moonrise"),
        "moonset": astro.get("moonset"),
        "moon_illumination": round(illum * 100) if illum is not None else None,
        "moon_phase_name": _moon_name(illum, waxing),
        "solunar_major": [t for t in (astro.get("moon_transit"), astro.get("moon_underfoot")) if t],
        "solunar_minor": [t for t in (astro.get("moonrise"), astro.get("moonset")) if t],
        "weather_source": used_source,
        "water_source": local_used.get("temp"),
    }


async def _apply_local_sensors(hass, hourly, temp_sensor, pressure_sensor, lat, lon, tz_name, elevation):
    """Replace air-temp-as-water and modelled pressure with better data.

    Temperature (used as water temperature): a physical sensor wins for today;
    otherwise estimate water temperature from the recent air temperature at the
    pond's own coordinates (applied to every day, since water temp is stable).
    Pressure: a local barometer's real recent trend overrides today's modelled
    trend. Returns (hourly, used) where used flags what was applied.
    """
    used = {"temp": False, "pressure": False}
    today = datetime.date.today()
    today_mask = hourly["date"] == today

    if temp_sensor:
        st = hass.states.get(temp_sensor)
        if st and st.state not in (None, "unknown", "unavailable", ""):
            val = _to_celsius(st.state, st.attributes.get("unit_of_measurement"))
            if val is not None and today_mask.any():
                hourly.loc[today_mask, "temp"] = val
                used["temp"] = "sensor"

    if not used["temp"]:
        est = await _water_temp_estimate(hass, lat, lon, tz_name, elevation)
        if est is not None:
            hourly["temp"] = est
            used["temp"] = "estimate"

    if pressure_sensor:
        try:
            trend = await _pressure_trend_from_history(hass, pressure_sensor, tz_name)
            if trend is not None and today_mask.any():
                hourly.loc[today_mask, "pressure_trend"] = trend
                used["pressure"] = True
        except Exception as e:
            _LOGGER.warning("Local pressure trend failed for %s: %s", pressure_sensor, e)

    return hourly, used


async def _pressure_trend_from_history(hass, sensor_id, tz_name):
    """Compute the pressure tendency (hPa/hour) from the last ~3h of a local
    barometer's recorder history."""
    from homeassistant.components.recorder import history, get_instance

    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = _tzutc.utc
    now = datetime.datetime.now(tz)
    start = now - datetime.timedelta(hours=3)

    states_dict = await get_instance(hass).async_add_executor_job(
        history.state_changes_during_period, hass, start, now, sensor_id
    )
    states = states_dict.get(sensor_id, []) if states_dict else []

    unit = None
    points = []
    for s in states:
        if s.state in (None, "unknown", "unavailable", ""):
            continue
        if unit is None:
            unit = s.attributes.get("unit_of_measurement")
        p = _to_hpa(s.state, unit or "hPa")
        changed = getattr(s, "last_changed", None)
        if p is None or changed is None:
            continue
        points.append((changed.timestamp(), p))

    if len(points) < 2:
        return None
    points.sort()
    (t0, p0), (t1, p1) = points[0], points[-1]
    hours = (t1 - t0) / 3600.0
    if hours < 0.5:
        return None
    return (p1 - p0) / hours


def _score_hour(row, profile, astro, weights: dict) -> float:
    hour = row["hour"]

    temp_score = _score_temp(row["temp"], profile["temp_range"])
    cloud = row["cloud"]
    cloud_score = 0.7 if pd.isna(cloud) else 1 - abs(cloud - profile["ideal_cloud"]) / 100
    press_score = _score_pressure_trend(
        row["pressure_trend"], profile.get("prefers_low_pressure", True)
    )
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

def _score_pressure_trend(trend: float, prefers_low: bool = True) -> float:
    # Fish that prefer low/falling pressure feed hard as a front approaches and
    # shut down on a rising barometer. Species that don't are the opposite:
    # they do best on a stable barometer. `prefers_low_pressure` per fish
    # (previously defined but unused) now actually drives this.
    if pd.isna(trend):
        return 0.7
    if prefers_low:
        if trend < -2:
            return 1.0
        elif trend > 2:
            return 0.4
        return 0.7
    # Species tolerant of / preferring stable or high pressure. Stable is the
    # common case, so keep it merely "good" (0.8) rather than perfect to avoid
    # pinning these species at the top every day.
    if trend > 2:
        return 0.6
    elif trend < -2:
        return 0.7
    return 0.8

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
    # `phase` is the Moon's illuminated fraction: 0.0 = new, 1.0 = full.
    # Solunar theory rates both the new and the full moon best, so reward both
    # extremes and score the quarters lower.
    if phase is None:
        return 0.7  # Default score when moon phase data is missing
    if phase < 0.15 or phase > 0.85:
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
