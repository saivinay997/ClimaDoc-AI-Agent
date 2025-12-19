from datetime import datetime, timedelta
from pymongo import MongoClient
from typing import Optional, Dict
from dotenv import load_dotenv
from secrets_loader import get_secret
load_dotenv()
import os
os.environ.setdefault("MONGO_URI", get_secret("MONGO_URI"))
MONGO_URI = os.getenv("MONGO_URI")

class MongoCache:
    def __init__(self, db_name: str = "ClimaDoc"):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[db_name]

        self.location_cache = self.db.location_cache
        self.weather_cache = self.db.weather_cache


    # ---------- Location Cache ----------

    def get_location(self, location_query: str) -> Optional[Dict]:
        return self.location_cache.find_one(
            {"location_query": {"$regex": location_query,"$options": "i"}},
            {"_id": 0}
        )

    def set_location(self, location_query: str, data: Dict):
        self.location_cache.update_one(
            {"location_query": location_query},
            {"$set": data},
            upsert=True
        )

    # ---------- Weather Cache ----------

    def get_weather(
        self,
        lat: float,
        lon: float,
        units: str
    ) -> Optional[Dict]:
        """
        Fetch cached weather using lat/lon.
        """
        record = self.weather_cache.find_one(
            {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "units": units
            },
            {"_id": 0, "data": 1}
        )

        return record["data"] if record else None

    def set_weather(
        self,
        lat: float,
        lon: float,
        units: str,
        data: Dict,
        ttl_minutes: int = 10
    ):
        """
        Cache weather data.
        """

        payload = {
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "units": units,
            "data": data,
            "updated_at": datetime.utcnow()
        }

        self.weather_cache.update_one(
            {
                "lat": payload["lat"],
                "lon": payload["lon"],
                "units": units
            },
            {"$set": payload},
            upsert=True
        )
