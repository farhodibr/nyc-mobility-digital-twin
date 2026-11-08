from pathlib import Path

import pandas as pd


def add_zone_metadata(
    df: pd.DataFrame,
    zone_lookup_path: str | Path = "data/external/taxi_zone_lookup.csv",
    location_id_col: str = "location_id",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """
    Add to zone-hour dataframe data with borough, zone name, service_zone.

    Args:
        df : dataframe with a zone/location ID column (e.g. the gold table)
        zone_lookup_path : path to the TLC zone lookup CSV
        location_id_col : name of df's zone-ID column
        drop_unmapped : if True, drops rows where LocationID is 264 (Unknown)
            or 265 (Outside of NYC) — these aren't real taxi zones, so they
            can't sensibly be forecast per-zone. If False, keeps them as-is.

    Returns:
        df with new columns: borough_name, zone_name, service_zone
    """
    df = df.copy()
    # load taxi_zone_lookup.csv with pd.read_csv
    zone_lookup_df = pd.read_csv(zone_lookup_path, keep_default_na=False)
    zone_lookup_df.columns = zone_lookup_df.columns.str.lower()

    zone_lookup_df = zone_lookup_df.rename(
        columns={"zone": "zone_name", "borough": "borough_name", "locationid": "location_id"}
    )
    # check df[location_id_col].dtype vs zone_lookup['LocationID'].dtype
    #       and cast one to match the other before merging
    if df[location_id_col].dtype != zone_lookup_df["location_id"].dtype:
        df[location_id_col] = df[location_id_col].astype(int)
        zone_lookup_df["location_id"] = zone_lookup_df["location_id"].astype(int)
    # left-merge df onto zone_lookup
    df_merged = df.merge(
        zone_lookup_df, left_on=location_id_col, right_on="location_id", how="left"
    )

    # if drop_unmapped, filters out rows where location_id is 264 or 265
    if drop_unmapped:
        df_merged = df_merged[~df_merged[location_id_col].isin([264, 265])]
    return df_merged
