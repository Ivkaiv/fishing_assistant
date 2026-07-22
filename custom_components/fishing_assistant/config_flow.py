from __future__ import annotations

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import DOMAIN
from .helpers.location import resolve_location_metadata_sync
from .fish_profiles import get_fish_species


def _weather_source_options(hass) -> list[dict]:
    """Build the list of selectable weather sources: Open-Meteo + weather.* entities."""
    options = [{"value": "open_meteo", "label": "Open-Meteo (model)"}]
    for state in sorted(hass.states.async_all("weather"), key=lambda s: s.entity_id):
        label = state.attributes.get("friendly_name") or state.entity_id
        options.append({"value": state.entity_id, "label": f"{label} ({state.entity_id})"})
    return options


def _default_weather_source(hass) -> str:
    """Default to a PirateWeather entity if present, else Open-Meteo."""
    for state in hass.states.async_all("weather"):
        if "pirateweather" in state.entity_id:
            return state.entity_id
    return "open_meteo"


def _temperature_sensor_selector():
    return selector.EntitySelector(
        selector.EntitySelectorConfig(domain="sensor", device_class="temperature")
    )


def _pressure_sensor_selector():
    return selector.EntitySelector(
        selector.EntitySelectorConfig(
            domain="sensor", device_class=["pressure", "atmospheric_pressure"]
        )
    )


class FishingAssistantConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle Fishing Assistant config flow."""

    VERSION = 2

    async def async_step_user(self, user_input=None):
        errors = {}

        if user_input is not None:
            name = user_input["name"]
            lat = user_input["latitude"]
            lon = user_input["longitude"]
            fish = user_input["fish"]
            body_type = user_input["body_type"]
            weather_source = user_input.get("weather_source", "open_meteo")

            await self.async_set_unique_id(f"{lat:.5f}_{lon:.5f}")
            self._abort_if_unique_id_configured()

            metadata = await self.hass.async_add_executor_job(
                resolve_location_metadata_sync, lat, lon
            )

            return self.async_create_entry(
                title=name,
                data={
                    "name": name,
                    "latitude": lat,
                    "longitude": lon,
                    "fish": fish,
                    "body_type": body_type,
                    "weather_source": weather_source,
                    "temperature_sensor": user_input.get("temperature_sensor"),
                    "pressure_sensor": user_input.get("pressure_sensor"),
                    "elevation": metadata.get("elevation"),
                    "timezone": metadata.get("timezone"),
                },
            )

        return self.async_show_form(
            step_id="user",
            data_schema=vol.Schema({
                vol.Required("name"): str,
                vol.Required("latitude"): vol.Coerce(float),
                vol.Required("longitude"): vol.Coerce(float),
                vol.Required("fish"): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": f, "label": f.replace("_", " ").title()}
                            for f in sorted(get_fish_species())
                        ],
                        multiple=True,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Required("body_type"): vol.In(["lake", "river", "pond", "reservoir"]),
                vol.Required("weather_source", default=_default_weather_source(self.hass)):
                    selector.SelectSelector(
                        selector.SelectSelectorConfig(
                            options=_weather_source_options(self.hass),
                            mode=selector.SelectSelectorMode.DROPDOWN,
                        )
                    ),
                vol.Optional("temperature_sensor"): _temperature_sensor_selector(),
                vol.Optional("pressure_sensor"): _pressure_sensor_selector(),
            }),
            errors=errors
        )

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return FishingAssistantOptionsFlow()
    
    @staticmethod
    @callback
    def async_get_entry_title(entry: ConfigEntry) -> str:
        """Return the title of the config entry shown in the UI."""
        return entry.data.get("name", "Fishing Location")



class FishingAssistantOptionsFlow(config_entries.OptionsFlow):
    """Handle options for Fishing Assistant."""

    # Note: don't set self.config_entry in __init__ — modern Home Assistant
    # exposes it as a read-only property set by the flow manager.

    async def async_step_init(self, user_input=None):
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema({
                vol.Required("fish", default=self.config_entry.data.get("fish", [])): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=[
                            {"value": f, "label": f.replace("_", " ").title()}
                            for f in sorted(get_fish_species())
                        ],
                        multiple=True,
                        mode=selector.SelectSelectorMode.DROPDOWN
                    )
                ),
                vol.Required("body_type", default=self.config_entry.data.get("body_type", "lake")):
                    vol.In(["lake", "river", "pond", "reservoir"]),
                vol.Required(
                    "weather_source",
                    default=self.config_entry.options.get(
                        "weather_source",
                        self.config_entry.data.get(
                            "weather_source", _default_weather_source(self.hass)
                        ),
                    ),
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(
                        options=_weather_source_options(self.hass),
                        mode=selector.SelectSelectorMode.DROPDOWN,
                    )
                ),
                vol.Optional(
                    "temperature_sensor",
                    description={"suggested_value": self.config_entry.options.get(
                        "temperature_sensor",
                        self.config_entry.data.get("temperature_sensor"),
                    )},
                ): _temperature_sensor_selector(),
                vol.Optional(
                    "pressure_sensor",
                    description={"suggested_value": self.config_entry.options.get(
                        "pressure_sensor",
                        self.config_entry.data.get("pressure_sensor"),
                    )},
                ): _pressure_sensor_selector(),
            })
        )
