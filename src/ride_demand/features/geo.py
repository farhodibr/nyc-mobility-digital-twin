import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates distance between two locations using longitudes and latitudes

    This calculation is more precise compared to Eucledian's distance
    formula. Real-world distance between two lat/lon points,
    accounting for Earth's curvature. Plain Euclidean distance on
    raw lat/lon degrees is invalid here — a degree of longitude represents
    a different real-world distance depending on latitude,
    so naive distance calculations distort increasingly
    the further from the equator you are.

    Args:
        lat1: latitude of first geopoint, e.g 40.65894
        lon1: longitude of first geopoint, e.g -74.65214
        lat2: latitude of second geopoint
        lon2: longitude of second geopoint

    Returns:
        Great-circle distance between the two points, in miles.
    """
    R = 3959  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c
