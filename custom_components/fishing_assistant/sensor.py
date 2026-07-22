from homeassistant.helpers.entity import Entity
from homeassistant.core import HomeAssistant
from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.components.sensor import SensorEntity, SensorDeviceClass
from .const import DOMAIN
import datetime

from .score import get_fish_score_forecast, scale_score


def default_weather_source(hass) -> str:
    """Pick a sensible default weather source.

    Prefer a PirateWeather entity if the user has one (full 7-day hourly
    forecast with all fields), otherwise fall back to the Open-Meteo model.
    """
    for state in hass.states.async_all("weather"):
        if "pirateweather" in state.entity_id:
            return state.entity_id
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
    weather_source = data.get("weather_source") or default_weather_source(hass)

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
                config_entry_id=config_entry.entry_id
            )
        )

    async_add_entities(sensors)


class FishScoreSensor(SensorEntity):
    should_poll = True
    
    def __init__(self, name, fish, lat, lon, body_type, timezone, elevation, config_entry_id, weather_source="open_meteo"):
        self._last_update_hour = None
        self._config_entry_id = config_entry_id
        self._weather_source = weather_source
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
            "weather_source": weather_source,
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

    async def async_update(self):
        """Fetch the 7-day forecast and set today's score as state."""
        now = datetime.datetime.now()
        update_hours = [0, 6, 12, 18]
    
        if self._last_update_hour is not None and now.hour not in update_hours:
            return

        if self._last_update_hour == now.hour:
            return

        forecast = await get_fish_score_forecast(
            hass=self.hass,
            fish=self._attrs["fish"],
            lat=self._attrs["lat"],
            lon=self._attrs["lon"],
            timezone=self._attrs["timezone"],
            elevation=self._attrs["elevation"],
            body_type=self._attrs["body_type"],
            weather_source=self._weather_source,
        )

        # Separate the meta block (which data source actually produced the
        # forecast) from the per-day data before storing it.
        meta = forecast.pop("meta", {})

        today_str = datetime.date.today().strftime("%Y-%m-%d")
        today_data = forecast.get(today_str, {})
        self._state = today_data.get("score", 0)

        self._attrs["forecast"] = forecast
        self._attrs["weather_source_active"] = meta.get("weather_source", self._weather_source)
        self._last_update_hour = now.hour

    async def async_added_to_hass(self):
        await self.async_update()