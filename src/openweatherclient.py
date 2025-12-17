import os
import requests
from typing import Dict, Optional
from datetime import datetime, timezone

from mongodb_cache import MongoCache
from utils import info_logger, unix_to_ist, local_time_to_unix
from secrets_loader import get_secret

OPENWEATHER_API_KEY = get_secret("OPENWEATHER_API_KEY")


class OpenWeatherGeoClient:
    BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(self, timeout: int = 10):
        if not OPENWEATHER_API_KEY:
            raise ValueError("OpenWeather API key is required")

        self.api_key = OPENWEATHER_API_KEY
        self.timeout = timeout
        self.cache = MongoCache()

    def get_coordinates(
        self,
        city: str,
        state_code: str | None = None,
        country_code: str | None = None
    ) -> Dict:
        """
        Resolve city/state/country to latitude and longitude.
        """
        location = city
        if state_code:
            location += f",{state_code}"
        if country_code:
            location += f",{country_code}"
        
       
        cached_data = self.cache.get_location(location)
        if cached_data:
            info_logger(f"Cache hit for location: {location}")
            data = cached_data
        else:

            params = {
                "q": location,
                "appid": self.api_key
            }

            response = requests.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout
            )
            if response.status_code != 200:
                raise Exception(f"Failed to get coordinates for `{location}`. Error: {response.json()}")
            data = response.json()
            self.cache.set_location(location, data)
            
        return data.get("coord", {})

class OpenWeatherClient:
    WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

    def __init__(
        self,
        timeout: int = 10,
        units: str = "metric",
        lang: str = "en"
    ):
        self.api_key = OPENWEATHER_API_KEY
        self.geo_client = OpenWeatherGeoClient()
        self.timeout = timeout
        self.units = units
        self.lang = lang
        self.cache = MongoCache()

    def _fetch_weather(self, lat: float, lon: float) -> Dict:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": self.api_key,
            "units": self.units,
            "lang": self.lang
        }

        response = requests.get(
            self.WEATHER_URL,
            params=params,
            timeout=self.timeout
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get weather for `{lat}, {lon}` with units `{self.units}`. Error: {response.json()}")
        return response.json()

    def get_weather_by_coordinates(self, lat: float, lon: float) -> Dict:
        """
        Direct lat/lon weather fetch.
        """
        return self._fetch_weather(lat, lon)

    def get_weather_by_location(
        self,
        city: str,
        state_code: str | None = None,
        country_code: str | None = None
    ) -> Dict:
        """
        Location-based weather using Geo wrapper → lat/lon → weather.
        """
        location = city
        if state_code:
            location += f",{state_code}"
        if country_code:
            location += f",{country_code}"
        
        coords = self.geo_client.get_coordinates(
            city=city,
            state_code=state_code,
            country_code=country_code
        )
        lat, lon = coords.get("lat"), coords.get("lon")
        
        cached_weather = self.cache.get_weather(lat=lat, lon=lon, units=self.units)
        if cached_weather:
            info_logger(f"Cache hit for weather: {location}")
            weather = cached_weather
            if weather.get("dt"):
                # check if the dt is not more than 30 mins
                cache_time =  60 * 30 # 30 mins
                print(weather["dt"])
                diff = abs(local_time_to_unix(weather["dt"]) - int(datetime.now(timezone.utc).timestamp()))
                if diff < cache_time:
                    info_logger(f"Cache hit for weather: {location}. Cached time difference is {diff} seconds")
                    return {
                            "location": {
                                "city": location,
                                "lat": lat,
                                "lon": lon,
                            },
                            "weather": weather
                        }
                else:
                    info_logger(f"Cache expired: {diff}")
        
        weather = self._fetch_weather(lat=lat, lon=lon)
        weather["sys"]["sunrise"] = unix_to_ist(weather["sys"]["sunrise"])
        weather["sys"]["sunset"] = unix_to_ist(weather["sys"]["sunset"])
        weather["dt"] = unix_to_ist(weather["dt"])
        print(weather["dt"])
        self.cache.set_weather(lat=lat, lon=lon, units=self.units, data=weather)
        
        return {
            "location": {
                "city": location,
                "lat": lat,
                "lon": lon,
            },
            "weather": weather
        }
        
