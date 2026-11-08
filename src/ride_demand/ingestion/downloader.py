import urllib.request
from pathlib import Path

# 4 levels up: ingestion/ -> ride_demand/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def download_tlc_file(dataset: str, year: int, month: int) -> Path:
    """Download one month of NYC TLC trip data, skipping if already cached.

    Fetches the requested monthly Parquet file from TLC's CloudFront
    distribution and saves it to the Bronze layer (data/bronze/).
    THe function checks If the file already exists locally,
    the download is skipped and the
    existing path is returned immediately.

    Args:
        dataset: TLC dataset prefix, here "fhvhv" for High Volume FHV.
        year: Four-digit year, e.g. 2021.
        month: Month as a plain integer (1-12); zero-padding is handled
            internally.

    Returns:
        Path to the local Parquet file, whether newly downloaded or
        already cached.
    """
    month_str = f"{month:02d}"
    filename = f"{dataset}_tripdata_{year}-{month_str}.parquet"
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
    path_to_save = PROJECT_ROOT / "data" / "bronze" / filename

    if not path_to_save.exists():
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path_to_save)

    return path_to_save
