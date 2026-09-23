import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path

# Setup the Open-Meteo API client with cache and retry on error

BASE_DIR = Path(__file__).resolve()
PROJECT_ROOT = BASE_DIR.parents[2]   # ajusta si tu estructura cambia
DATA_DIR = PROJECT_ROOT / "datos" / "limpios"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DATA_DIR / "nyc_weather_2023_first_half.csv"

CACHE_DIR = PROJECT_ROOT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Crear el archivo de cache en esa carpeta
cache_file = CACHE_DIR / "openmeteo_cache.sqlite"
cache_session = requests_cache.CachedSession(str(cache_file), expire_after=-1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 40.7143,
	"longitude": -74.006,
	"start_date": "2023-01-01",
	"end_date": "2023-12-31",
	"hourly": ["temperature_2m", "precipitation", "rain", "snowfall", "snow_depth"],
    "timezone": "America/New_York"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
hourly_snow_depth = hourly.Variables(4).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["snowfall"] = hourly_snowfall
hourly_data["snow_depth"] = hourly_snow_depth

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

hourly_dataframe.to_csv(OUTPUT_FILE, index=False)

df = pd.read_csv(OUTPUT_FILE, parse_dates=["date"])

df["day"] = df["date"].dt.day

df_first_week = df[df["day"] <= 14].copy()

df_first_week.drop(columns=["day"], inplace=True)

print(df_first_week.head(20))

df_first_week.to_csv(OUTPUT_FILE, index=False)
