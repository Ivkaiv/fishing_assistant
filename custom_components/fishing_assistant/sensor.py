from homeassistant.helpers.entity import Entity
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.components.sensor import SensorEntity, SensorDeviceClass
from homeassistant.const import EVENT_HOMEASSISTANT_STARTED
from homeassistant.helpers.event import (
    async_track_time_change,
    async_track_state_change_event,
    async_track_time_interval,
)
from .const import DOMAIN
import datetime

from .score import get_fish_score_forecast, scale_score


def default_weather_source(hass) -> str:
    """Pick a sensible default weather source, resolved lazily at fetch time.

    Prefer coordinate-based PirateWeather (queried at the pond's own lat/lon)
    when the user has a PirateWeather API key, otherwise the coordinate-based
    Open-Meteo model.
    """
    try:
        for entry in hass.config_entries.async_entries("pirateweather"):
            if entry.data.get("api_key"):
                return "pirateweather"
    except Exception:
        pass
    return "open_meteo"


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: ConfigEntry,
    async_add_entities
):
    """Set up fishing assistant sensors from a config entry."""
    # Options (set via the reconfigure dialog) override the original data.
    data = {**config_entry.data, **config_entry.options}
    sensors = []

    name = data["name"]
    lat = data["latitude"]
    lon = data["longitude"]
    fish_list = data["fish"]
    body_type = data["body_type"]
    timezone = data["timezone"]
    elevation = data["elevation"]
    # Keep the raw configured value (which may be absent for locations created
    # before this option existed). The default is resolved lazily at fetch
    # time, because the chosen weather integration can load AFTER us during
    # startup — resolving it once here would freeze it to Open-Meteo.
    weather_source = data.get("weather_source")
    temp_sensor = data.get("temperature_sensor")
    pressure_sensor = data.get("pressure_sensor")

    for fish in fish_list:
        sensors.append(
            FishScoreSensor(
                name=name,
                fish=fish,
                lat=lat,
                lon=lon,
                timezone=timezone,
                body_type=body_type,
                elevation=elevation,
                weather_source=weather_source,
                temp_sensor=temp_sensor,
                pressure_sensor=pressure_sensor,
                config_entry_id=config_entry.entry_id
            )
        )

    async_add_entities(sensors)


class FishScoreSensor(SensorEntity):
    should_poll = False

    def __init__(self, name, fish, lat, lon, body_type, timezone, elevation, config_entry_id,
                 weather_source=None, temp_sensor=None, pressure_sensor=None):
        self._config_entry_id = config_entry_id
        self._weather_source = weather_source
        self._temp_sensor = temp_sensor
        self._pressure_sensor = pressure_sensor
        self._device_identifier = f"{name}_{lat}_{lon}"
        self._name = f"{name.lower().replace(' ', '_')}_{fish}_score"
        self._friendly_name = f"{name} ({fish.title()}) Fishing Score"
        self._state = None
        self._attrs = {
            "fish": fish,
            "location": name,
            "lat": lat,
            "lon": lon,
            "body_type": body_type,
            "timezone": timezone,
            "elevation": elevation,
            "weather_source": weather_source or "auto",
            "temperature_sensor": temp_sensor,
            "pressure_sensor": pressure_sensor,
        }

    @property
    def name(self):
        return self._friendly_name

    @property
    def unique_id(self):
        return self._name

    @property
    def device_class(self):
        return None  

    @property
    def entity_category(self):
        return None  
    
    @property
    def icon(self):
        return "mdi:fish"
    
    @property
    def native_value(self):
        return self._state

    @property
    def extra_state_attributes(self):
        return self._attrs
    
    @property
    def device_info(self):
        return {
            "identifiers": {(DOMAIN, self._device_identifier)},
            "name": self._attrs["location"],
            "manufacturer": "Fishing Assistant",
            "model": "Fish Score Sensor",
            "entry_type": "service",
            "via_device": None            
        }

    def _effective_source(self) -> str:
        """Resolve the weather source to use right now. An unset/'auto' config
        resolves to the default (PirateWeather if available, else Open-Meteo)
        every time, so it upgrades once the weather integration has loaded."""
        if self._weather_source in (None, "", "auto"):
            return default_weather_source(self.hass)
        return self._weather_source

    async def async_update(self):
        """Fetch the 7-day forecast and set today's score as state."""
        forecast = await get_fish_score_forecast(
            hass=self.hass,
            fish=self._attrs["fish"],
            lat=self._attrs["lat"],
            lon=self._attrs["lon"],
            timezone=self._attrs["timezone"],
            elevation=self._attrs["elevation"],
            body_type=self._attrs["body_type"],
            weather_source=self._effective_source(),
            temp_sensor=self._temp_sensor,
            pressure_sensor=self._pressure_sensor,
        )

        # Separate the meta block (which data source actually produced the
        # forecast) from the per-day data before storing it.
        meta = forecast.pop("meta", {})
        active = meta.get("weather_source", self._weather_source)

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_data = forecast.get(today_str, {})
        if forecast:
            self._state = today_data.get("score", 0)
            self._attrs["forecast"] = forecast
            self._attrs["weather_source_active"] = active
            self._attrs["local_temp_used"] = meta.get("local_temp", False)
            self._attrs["local_pressure_used"] = meta.get("local_pressure", False)

    async def _scheduled_update(self, _now):
        """Recompute at the scheduled refresh times (0/6/12/18)."""
        await self.async_update()
        self.async_write_ha_state()

    async def _first_update(self, _event):
        await self.async_update()
        self.async_write_ha_state()

    async def _on_source_changed(self, _event):
        """Recompute when the chosen weather entity posts an update, but only
        while we are still on the Open-Meteo fallback. This upgrades to the
        preferred source as soon as it becomes ready after a restart, without
        polling, and stops once we're already using it."""
        if self._attrs.get("weather_source_active") in (None, "open_meteo"):
            await self.async_update()
            self.async_write_ha_state()

    async def _maybe_upgrade(self, _now):
        """Every few minutes, retry the preferred weather source while we are
        still on the Open-Meteo fallback (e.g. the chosen weather integration
        loaded after us, or its forecast wasn't ready yet right after a
        restart). No work once already upgraded."""
        eff = self._effective_source()
        wants_entity = eff not in ("open_meteo", "", None)
        if wants_entity and self._attrs.get("weather_source_active") in (None, "open_meteo"):
            await self.async_update()
            self.async_write_ha_state()

    async def async_added_to_hass(self):
        # Refresh the forecast four times a day.
        self.async_on_remove(
            async_track_time_change(
                self.hass, self._scheduled_update, hour=[0, 6, 12, 18], minute=1, second=0
            )
        )
        # Unless the user explicitly pinned Open-Meteo, keep trying to reach the
        # preferred weather entity: recompute when it posts an update, and retry
        # every 5 minutes as a guaranteed fallback (covers the entity loading
        # after us, or a slow/rate-limited forecast right after a restart).
        if self._weather_source != "open_meteo":
            eff = self._effective_source()
            # Only HA weather.* sources are entities we can watch for updates.
            if eff and str(eff).startswith("weather."):
                self.async_on_remove(
                    async_track_state_change_event(
                        self.hass, [eff], self._on_source_changed
                    )
                )
            self.async_on_remove(
                async_track_time_interval(
                    self.hass, self._maybe_upgrade, datetime.timedelta(minutes=5)
                )
            )
        # Compute once now, but wait until HA has fully started so the chosen
        # weather entity is set up before the first fetch.
        if self.hass.is_running:
            await self.async_update()
        else:
            self.hass.bus.async_listen_once(EVENT_HOMEASSISTANT_STARTED, self._first_update)