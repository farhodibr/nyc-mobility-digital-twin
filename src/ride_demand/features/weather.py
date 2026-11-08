import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def download_weather(location_id: int, centroid: Point, start_date: str, stop_date: str) -> Path:
    """
    Download weather data from Open Meteo API, skipping
    if already exists in the downloaded files.
    Time period is selected by user.

    Args:
        location_id: Location ID of TLC's zone.
        centroid: The zone's centroid as a shapely Point (accessed via
        .x for longitude, .y for latitude).
        start_date: Start date for weather data. Format YYYY-MM-DD in string.
        stop_date: Stop date for weather data. Format YYYY-MM-DD in string.

    Returns:
            Path to the local JSON file, whether newly downloaded or
            already downloaded.
    """

    filename = f"zone_{location_id}_weather_{start_date}_to_{stop_date}.json"
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={centroid.y}&longitude={centroid.x}&start_date={start_date}&end_date={stop_date}&hourly=temperature_2m,precipitation,rain,snowfall,wind_speed_10m"
    path_to_save = PROJECT_ROOT / "data" / "external" / "weather" / filename

    if not path_to_save.exists():
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path_to_save)

    return path_to_save


def download_weather_with_retry(
    location_id: int, centroid: Point, start_date: str, stop_date: str, max_retries: int = 5
) -> Path:
    """
    This function adds resilince against Open Meteo's rate limiting.
    It wraps download_weather and gives 2 seconds breaks
    between calls (even after succesfull calls). Maximum retries are set to 5 by default.
    If fetching error's code is specifically 429, it waits
    and retries - with the wait growing longer each attempt.
    Other kinds of errors (not 429) are not retried — they propagate immediately.
    If it exhausts all retries, it raises its own Exception.

    Args:
            location_id: Location ID of TLC's zone.
            centroid: The zone's centroid as a shapely Point (accessed via
            .x for longitude, .y for latitude).
            start_date: Start date for weather data. Format YYYY-MM-DD in string.
            stop_date: Stop date for weather data. Format YYYY-MM-DD in string.
            max_retries: Number's of retries for error 429. Default atempts number is 5.

     Returns:
        Path to the local JSON file, whether newly downloaded or
        already cached.

    Raises:
        Exception: If all max_retries attempts are exhausted without success.
        urllib.error.HTTPError: If a non-429 HTTP error occurs
        (not retried, propagated immediately).
    """
    for attempt in range(max_retries):
        try:
            result = download_weather(location_id, centroid, start_date, stop_date)
            time.sleep(2)  # throttles successful calls
            return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (attempt + 1)
                print(
                    f"Zone {location_id}: rate limited, waiting {wait}s "
                    f"(attempt {attempt+1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                raise
    raise Exception(f"Zone {location_id} failed after {max_retries} retries")


def combine_weather_files(location_ids: list[int], start_date: str, stop_date: str) -> pd.DataFrame:
    """
    Function converts JSON weather files into panda dfs and combines all dfs into single dataframe.

    Args:
        location_ids: list of TLC zone location IDs
        start_date: start date of weather data
        stop_date: stop date of weather data

    Returns:
        DataFrame with one row per zone-hour,
        columns: time (datetime), temperature_2m, precipitation, rain, snowfall, wind_speed_10m
        (all float), and location_id (int).
    """
    weather_dfs_combined = []

    for location_id in location_ids:
        try:
            with open(
                PROJECT_ROOT
                / "data"
                / "external"
                / "weather"
                / f"zone_{location_id}_weather_{start_date}_to_{stop_date}.json"
            ) as f:
                data = json.load(f)
                data_df = pd.DataFrame(data["hourly"])
                data_df["time"] = pd.to_datetime(data_df["time"])
                data_df["location_id"] = location_id
                weather_dfs_combined.append(data_df)
        except FileNotFoundError as e:
            print(f"No file found for location {location_id}: {e}")

    all_weather = pd.concat(weather_dfs_combined, ignore_index=True)
    return all_weather
