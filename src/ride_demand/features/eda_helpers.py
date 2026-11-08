from pathlib import Path

import pandas as pd

from ride_demand.features.zones import add_zone_metadata


def pick_comparison_zones(
    zone_totals: pd.DataFrame,
    zone_lookup_path: str | Path = "data/external/taxi_zone_lookup.csv",
    total_col: str = "total_pickups",
    jfk_zone_id: int = 132,
    quiet_percentile: float = 0.10,
) -> dict[str, int]:
    """
    Pick 3 zones for the target-distribution comparison histogram:
    JFK, the busiest Manhattan zone, and a quiet-but-real outer-borough
    zone.

    Parameters
    ----------
    zone_totals : one row per zone — location_id and total_col, already
        aggregated (not raw zone-hour rows)
    zone_lookup_path : passed through to add_zone_metadata()
    total_col : name of the pre-aggregated total-pickups column
    jfk_zone_id : JFK's location_id — fixed, no computation needed
    quiet_percentile : rank-position percentile for the "quiet" pick,
        among outer-borough, non-airport zones

    Returns
    -------
    dict, e.g. {'jfk': 132, 'busy_manhattan': ..., 'quiet_outer_borough': ...}
    """
    zone_totals = zone_totals.copy()

    zone_totals_with_boroughs = add_zone_metadata(
        zone_totals, zone_lookup_path=zone_lookup_path, drop_unmapped=True
    )

    idx = zone_totals_with_boroughs[zone_totals_with_boroughs["borough_name"] == "Manhattan"][
        total_col
    ].idxmax()

    busiest_zone_manhattan_loc_id = zone_totals_with_boroughs["location_id"].loc[idx]

    outer_zones = zone_totals_with_boroughs[
        (zone_totals_with_boroughs["borough_name"] != "Manhattan")
        & (zone_totals_with_boroughs["service_zone"] != "Airports")
    ]

    outer_sorted = outer_zones.sort_values(total_col)
    target_position = round(quiet_percentile * (len(outer_sorted) - 1))
    quiet_percentile_zone_outer_borough = outer_sorted["location_id"].iloc[target_position]

    zones = {
        "jfk": jfk_zone_id,
        "busy_manhattan": busiest_zone_manhattan_loc_id,
        "quiet_outer_borough": quiet_percentile_zone_outer_borough,
    }

    return zones
