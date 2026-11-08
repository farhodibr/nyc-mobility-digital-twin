"""Helpers for 03_mta_disruptions: coverage audit, rule-based episode
metadata, and leakage-safe as-of disruption features.

Drop into src/ (or paste cells) and call from the notebook. Every number
is computed by your own run; nothing here is precomputed.
"""

from __future__ import annotations

import re

import pandas as pd


# ---------------------------------------------------------------------------
# 1. Bronze coverage audit: catch missing days before anything downstream
# ---------------------------------------------------------------------------
def audit_daily_coverage(alerts: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar day in the covered range, with alert counts.

    Days with zero rows are almost certainly download/archive gaps, not
    "no disruptions" days, and they silently become false zeros in features.
    """
    days = alerts["date"].dt.normalize()
    full_range = pd.date_range(days.min(), days.max(), freq="D")
    daily = (
        days.value_counts()
        .reindex(full_range, fill_value=0)
        .rename_axis("day")
        .reset_index(name="alert_updates")
    )
    daily["is_gap"] = daily["alert_updates"].eq(0)
    return daily


# ---------------------------------------------------------------------------
# 2. Rule-based metadata (replaces most of the LLM step)
# ---------------------------------------------------------------------------
# Ordered: first match wins. Extend these lists from the "unmatched" audit.
CAUSE_PATTERNS: list[tuple[str, str]] = [
    ("signal_problem", r"\bsignal (?:problem|malfunction|issue)"),
    ("switch_problem", r"\bswitch (?:problem|malfunction|issue)"),
    ("power_problem", r"\b(?:power|electrical) (?:problem|issue|outage)"),
    ("track_condition", r"\b(?:track|rail) (?:condition|problem|defect|repair)|\bbroken rail"),
    ("fire_or_smoke", r"\bFDNY\b|\bfire\b|\bsmoke\b"),
    ("medical_emergency", r"\bEMS\b|\bmedical\b|\bsick (?:customer|person)|\bsomeone who needs medical"),
    ("person_on_tracks", r"\b(?:person|someone|customer) (?:on|struck|hit)\b.*\btracks?\b|\bunauthorized person"),
    ("police_activity", r"\bNYPD\b|\bpolice\b"),
    ("train_mechanical_problem", r"\b(?:mechanical|brake|braking|train with (?:a )?(?:problem|mechanical))"),
    ("door_problem", r"\bdoor (?:problem|malfunction|issue)"),
    ("debris_or_obstruction", r"\bdebris\b|\bobstruction\b|\bobject on\b"),
    ("weather", r"\b(?:weather|flooding|snow|ice|storm|heat)\b"),
    ("work_equipment", r"\bwork (?:train|equipment)\b"),
    ("staffing_or_crew", r"\b(?:crew|staffing|operator availability)\b"),
    ("cleaning", r"\bcleaning\b|\bclean(?:ing)? (?:a|the) train\b"),
    ("planned_maintenance", r"\bplanned work\b|\bscheduled (?:work|maintenance)\b"),
]

# "... at 36 St (BKLYN)." / "... near Chauncey St." -> station after at/near
CAUSE_STATION = re.compile(
    r"\b(?:address|addressed|respond(?:s|ed)? to|investigate|investigated)"
    r"\b[^.\n]*?\b(?:at|near)\s+([A-Z0-9][^.\n,]*?)(?:\.|,|\n|$)"
)
# A segment endpoint stops at punctuation or at the next clause.
_END = r"(?=\s+(?:while|after|because|as|in both|due)\b|[.,;\n]|$)"
SEGMENT_FROM_TO = re.compile(
    r"\bfrom\s+([A-Z0-9][^.\n]*?)\s+to\s+([A-Z0-9][^.\n]*?)" + _END
)
SEGMENT_BETWEEN = re.compile(
    r"\bbetween\s+([A-Z0-9][^.\n]*?)\s+and\s+([A-Z0-9][^.\n]*?)" + _END
)
# MTA writes directions as compass, uptown/downtown, or "<terminal>-bound".
DIRECTION = re.compile(
    r"\b(northbound|southbound|eastbound|westbound|uptown|downtown|"
    r"in both directions)\b|([A-Z][A-Za-z0-9 .'/&-]*?-bound)\b",
    flags=re.IGNORECASE,
)


def _text(*parts) -> str:
    return " ".join(str(p) for p in parts if p is not None and not pd.isna(p))


def extract_rule_metadata(header, description=None) -> dict:
    """Extract cause, locations and direction text from one alert update."""
    text = re.sub(r"[ \t]+", " ", _text(header, description))

    cause = next(
        (name for name, pat in CAUSE_PATTERNS if re.search(pat, text, re.IGNORECASE)),
        "unknown",
    )

    cause_station = CAUSE_STATION.search(text)
    segment = SEGMENT_FROM_TO.search(text) or SEGMENT_BETWEEN.search(text)

    directions = []
    for compass_or_word, terminal_bound in DIRECTION.findall(text):
        value = (compass_or_word or terminal_bound).strip()
        if value and value.lower() not in {d.lower() for d in directions}:
            directions.append(value)

    return {
        "cause": cause,
        "cause_station_text": cause_station.group(1).strip() if cause_station else None,
        "from_station_text": segment.group(1).strip() if segment else None,
        "to_station_text": segment.group(2).strip() if segment else None,
        "direction_texts": tuple(directions) or ("unspecified",),
    }


def episode_rule_metadata(
    episodes: pd.DataFrame,
    event_history: pd.DataFrame,
    *,
    context_updates_before_onset: int = 1,
) -> pd.DataFrame:
    """Metadata per episode using ONLY updates published at or before onset.

    Using the onset (plus pre-onset context) keeps the metadata knowable at
    onset time. Cause usually appears in the pre-onset 'delays' update, the
    segment in the onset update.
    """
    history = event_history.sort_values(["event_id", "date", "update_number", "alert_id"])
    rows = []

    for ep in episodes.itertuples(index=False):
        ev = history[history["event_id"].eq(ep.event_id)].reset_index(drop=True)
        onset_pos = ev.index[ev["alert_id"].eq(ep.onset_alert_id)][0]
        window = ev.iloc[max(0, onset_pos - context_updates_before_onset): onset_pos + 1]

        # Onset update first so its segment/direction wins; context fills gaps.
        merged: dict = {}
        for upd in window.iloc[::-1].itertuples(index=False):
            found = extract_rule_metadata(upd.header, upd.description)
            for key, value in found.items():
                empty = value in (None, "unknown", ("unspecified",))
                if key not in merged or (merged[key] in (None, "unknown", ("unspecified",)) and not empty):
                    merged[key] = value

        merged["event_id"] = ep.event_id
        merged["episode_number"] = ep.episode_number
        merged["needs_llm_or_review"] = (
            merged["cause"] == "unknown"
            or (merged["from_station_text"] is None and merged["cause_station_text"] is None)
        )
        rows.append(merged)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Leakage-safe features: state of every event AS OF each forecast origin T
# ---------------------------------------------------------------------------
PRIMARY_OPERATIONAL = {
    "severe_delay",
    "partial_suspension",
    "full_suspension",
    "stop_skipping",
    "reroute",
}


def route_disruption_state_asof(
    timeline: pd.DataFrame,
    origins: pd.DatetimeIndex,
    *,
    effects_column: str = "explicit_high_impact_effects",
    max_state_age: pd.Timedelta = pd.Timedelta(hours=3),
) -> pd.DataFrame:
    """Long table (origin T, route) of disruptions active at T.

    Uses, for each event, only its latest update with date <= T. An event
    whose last-known state is high-impact but has had no update for
    `max_state_age` is treated as stale (otherwise right-censored episodes
    stay "active" forever).
    """
    t = timeline.sort_values("date")[["event_id", "date", "affected", effects_column]].copy()
    t["is_primary"] = t[effects_column].apply(lambda e: any(x in PRIMARY_OPERATIONAL for x in e))
    t["is_severe"] = t[effects_column].apply(
        lambda e: any(x in {"full_suspension", "partial_suspension"} for x in e)
    )

    out = []
    for T in origins:
        known = t[t["date"].le(T)]
        latest = known.drop_duplicates("event_id", keep="last")
        live = latest[latest["is_primary"] & (T - latest["date"]).le(max_state_age)].copy()
        if live.empty:
            continue
        live["route"] = live["affected"].str.split("|")
        live = live.explode("route")
        live["route"] = live["route"].str.strip()
        agg = live.groupby("route").agg(
            active_events=("event_id", "nunique"),
            active_suspension_events=("is_severe", "sum"),
            minutes_since_latest_update=("date", lambda d: (T - d.max()).total_seconds() / 60),
        )
        agg["origin"] = T
        out.append(agg.reset_index())

    if not out:
        return pd.DataFrame(columns=["origin", "route", "active_events",
                                     "active_suspension_events", "minutes_since_latest_update"])
    return pd.concat(out, ignore_index=True)[
        ["origin", "route", "active_events", "active_suspension_events", "minutes_since_latest_update"]
    ]
