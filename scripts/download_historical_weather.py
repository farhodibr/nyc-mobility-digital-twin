#!/usr/bin/env python
"""Resumable Open-Meteo ECMWF single-run archive downloader.

Place this file in ``nyc-ride-demand-platform/scripts/`` and run it from the
project environment. It reads the weather request-class and required-run
manifests created by the data-engineering notebook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


API_URL = "https://customer-single-runs-api.open-meteo.com/v1/forecast"
RUN_MANIFEST_NAME = (
    "ecmwf_ifs_required_runs_20240810_20260630_"
    "availability_7h_fh19.parquet"
)
CLASS_MANIFEST_NAME = "nyc_weather_request_class_manifest.parquet"
RATE_STATE_NAME = "open_meteo_rate_state.json"


class RateLimitReached(RuntimeError):
    """Raised when Open-Meteo returns HTTP 429."""

    def __init__(self, retry_after_seconds: float | None = None) -> None:
        super().__init__("Open-Meteo returned HTTP 429")
        self.retry_after_seconds = retry_after_seconds


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def find_project_root(script_path: Path) -> Path:
    """Find the repository root from the script location or current folder."""
    candidates = [script_path.parent, *script_path.parents, Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        manifest = (
            candidate
            / "data"
            / "silver"
            / "weather_forecasts"
            / "open_meteo_single_runs"
            / "ecmwf_ifs"
            / "manifests"
            / RUN_MANIFEST_NAME
        )
        if manifest.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the project root. Place this script under the project "
        "directory or pass --project-root."
    )


def atomic_write_json(document: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(document, file, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(destination)


def atomic_write_manifest(manifest: pd.DataFrame, destination: Path) -> None:
    temporary = destination.with_suffix(".tmp.parquet")
    manifest.to_parquet(temporary, index=False)
    temporary.replace(destination)


def normalize_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def validate_stored_document(document: dict[str, Any], expected_run: pd.Timestamp) -> None:
    stored_run = normalize_utc(document["model_run_utc"])
    responses = document["responses"]
    representatives = document["representatives"]
    if stored_run != expected_run:
        raise ValueError(f"Stored run {stored_run} does not match {expected_run}.")
    if len(responses) != 28 or len(representatives) != 28:
        raise ValueError("Stored document does not contain 28 representatives.")
    if not all(len(payload["hourly"]["time"]) == 19 for payload in responses):
        raise ValueError("Stored document contains an incomplete hourly response.")


@dataclass
class RatePacer:
    state_path: Path
    hourly_run_budget: int = 150
    daily_run_budget: int = 300
    location_units_per_run: int = 28
    safety_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.events: list[float] = []
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.events = [float(item) for item in state.get("request_times", [])]
            except (json.JSONDecodeError, TypeError, ValueError):
                self.events = []
        self._prune()

    @property
    def hourly_unit_budget(self) -> int:
        return self.hourly_run_budget * self.location_units_per_run

    @property
    def daily_unit_budget(self) -> int:
        return self.daily_run_budget * self.location_units_per_run

    def _prune(self) -> None:
        cutoff = time.time() - 86_400
        self.events = [event for event in self.events if event > cutoff]

    def _save(self) -> None:
        self._prune()
        atomic_write_json(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "location_units_per_request": self.location_units_per_run,
                "hourly_run_budget": self.hourly_run_budget,
                "daily_run_budget": self.daily_run_budget,
                "request_times": self.events,
            },
            self.state_path,
        )

    def required_wait_seconds(self) -> float:
        self._prune()
        now = time.time()
        hourly_events = [event for event in self.events if event > now - 3_600]
        daily_events = self.events
        waits = [0.0]
        if len(hourly_events) >= self.hourly_run_budget:
            waits.append(hourly_events[0] + 3_600 + self.safety_seconds - now)
        if len(daily_events) >= self.daily_run_budget:
            waits.append(daily_events[0] + 86_400 + self.safety_seconds - now)
        return max(0.0, max(waits))

    def wait_for_capacity(self) -> None:
        while True:
            wait_seconds = self.required_wait_seconds()
            if wait_seconds <= 0:
                return
            resume_at = datetime.fromtimestamp(
                time.time() + wait_seconds, tz=timezone.utc
            )
            print(
                f"Rate budget reached. Waiting {wait_seconds / 60:.1f} minutes; "
                f"resume at {resume_at:%Y-%m-%d %H:%M:%S UTC}.",
                flush=True,
            )
            time.sleep(wait_seconds)

    def record_request(self) -> None:
        self.events.append(time.time())
        self._save()


def build_request_params(
    model_run_utc: pd.Timestamp, representatives: pd.DataFrame
) -> dict[str, Any]:
    return {
        "latitude": ",".join(representatives["request_latitude"].astype(str)),
        "longitude": ",".join(representatives["request_longitude"].astype(str)),
        "run": model_run_utc.strftime("%Y-%m-%dT%H:%M"),
        "models": "ecmwf_ifs",
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "rain",
                "snowfall",
                "wind_speed_10m",
            ]
        ),
        "forecast_hours": 19,
        "timezone": "UTC",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "apikey": os.environ["OPEN_METEO_API_KEY"],
    }


def request_with_server_retries(
    session: requests.Session,
    params: dict[str, Any],
    pacer: RatePacer,
    timeout_seconds: float,
) -> requests.Response:
    retry_delays = [0, 2, 5, 10]
    last_error: Exception | None = None
    for attempt, delay in enumerate(retry_delays, start=1):
        if delay:
            time.sleep(delay)
        pacer.wait_for_capacity()
        pacer.record_request()
        try:
            response = session.get(API_URL, params=params, timeout=timeout_seconds)
        except requests.RequestException as error:
            last_error = error
            if attempt == len(retry_delays):
                raise
            continue
        if response.status_code == 429:
            retry_header = response.headers.get("Retry-After")
            retry_after = float(retry_header) if retry_header else None
            raise RateLimitReached(retry_after)
        if response.status_code in {500, 502, 503, 504}:
            last_error = requests.HTTPError(
                f"Temporary server response {response.status_code}", response=response
            )
            if attempt == len(retry_delays):
                response.raise_for_status()
            continue
        response.raise_for_status()
        return response
    if last_error is not None:
        raise last_error
    raise RuntimeError("Request loop ended unexpectedly.")


def mark_complete(
    manifest: pd.DataFrame,
    index: int,
    status_code: int = 200,
) -> None:
    manifest.loc[index, "run_status"] = "complete"
    manifest.loc[index, "status_code"] = status_code
    manifest.loc[index, "returned_classes"] = 28
    manifest.loc[index, "returned_hours_per_class"] = 19
    manifest.loc[index, "downloaded_at_utc"] = utc_now()
    manifest.loc[index, "error_reason"] = pd.NA


def download_run(
    *,
    manifest: pd.DataFrame,
    index: int,
    project_root: Path,
    manifest_path: Path,
    representatives: pd.DataFrame,
    session: requests.Session,
    pacer: RatePacer,
    timeout_seconds: float,
) -> str:
    model_run_utc = normalize_utc(manifest.loc[index, "model_run_utc"])
    raw_path = project_root / Path(manifest.loc[index, "raw_json_relative_path"])

    if raw_path.exists():
        document = json.loads(raw_path.read_text(encoding="utf-8"))
        validate_stored_document(document, model_run_utc)
        mark_complete(manifest, index)
        atomic_write_manifest(manifest, manifest_path)
        return "recovered"

    manifest.loc[index, "attempt_count"] += 1
    params = build_request_params(model_run_utc, representatives)

    try:
        response = request_with_server_retries(
            session, params, pacer, timeout_seconds
        )
        payloads = response.json()
        if isinstance(payloads, dict):
            payloads = [payloads]
        if len(payloads) != 28:
            raise ValueError(f"Expected 28 payloads; received {len(payloads)}.")
        if not all(len(item["hourly"]["time"]) == 19 for item in payloads):
            raise ValueError("At least one payload did not contain 19 hours.")

        document = {
            "source": "Open-Meteo Single Runs API",
            "weather_model": "ecmwf_ifs",
            "model_run_utc": model_run_utc.isoformat(),
            "availability_delay_hours": 7,
            "forecast_hours": 19,
            "retrieved_at_utc": utc_now().isoformat(),
            "request_url": API_URL,
            "request_parameters": {
                key: value for key, value in params.items()
                if key != "apikey"
            },
            "representatives": representatives.to_dict(orient="records"),
            "responses": payloads,
        }
        atomic_write_json(document, raw_path)
        mark_complete(manifest, index, response.status_code)
        atomic_write_manifest(manifest, manifest_path)
        return "downloaded"

    except RateLimitReached:
        manifest.loc[index, "run_status"] = "pending"
        manifest.loc[index, "status_code"] = 429
        manifest.loc[index, "error_reason"] = (
            "rate_limit_429: queued for automatic retry"
        )
        atomic_write_manifest(manifest, manifest_path)
        raise
    except Exception as error:
        manifest.loc[index, "run_status"] = "failed"
        manifest.loc[index, "error_reason"] = (
            f"{type(error).__name__}: {error}"
        )[:1000]
        atomic_write_manifest(manifest, manifest_path)
        raise


def seconds_until_next_utc_day(buffer_minutes: int = 10) -> float:
    now = datetime.now(timezone.utc)
    tomorrow = pd.Timestamp(now).normalize() + pd.Timedelta(days=1)
    resume = tomorrow + pd.Timedelta(minutes=buffer_minutes)
    return max(60.0, (resume.to_pydatetime() - now).total_seconds())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument(
        "--run-manifest-name",
        default=RUN_MANIFEST_NAME,
        help=(
            "Run-manifest filename inside the ECMWF manifest directory. "
            "Defaults to the original historical manifest."
            ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=300,
        help="Maximum successful/recovered runs in this invocation (default: 300).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Continue across daily windows until every pending run is processed.",
    )
    parser.add_argument("--hourly-runs", type=int, default=100_000)
    parser.add_argument("--daily-runs", type=int, default=100_000)
    parser.add_argument("--pause-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--stop-on-429",
        action="store_true",
        help="Exit cleanly on 429 instead of waiting automatically.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = (
        args.project_root.resolve()
        if args.project_root
        else find_project_root(Path(__file__).resolve())
    )
    manifest_dir = (
        project_root
        / "data"
        / "silver"
        / "weather_forecasts"
        / "open_meteo_single_runs"
        / "ecmwf_ifs"
        / "manifests"
    )
    manifest_path = manifest_dir / args.run_manifest_name
    class_manifest_path = manifest_dir / CLASS_MANIFEST_NAME
    rate_state_path = manifest_dir / RATE_STATE_NAME

    manifest = pd.read_parquet(manifest_path)
    manifest["downloaded_at_utc"] = pd.to_datetime(
        manifest["downloaded_at_utc"], utc=True
    )
    representatives = (
        pd.read_parquet(class_manifest_path)
        .loc[lambda frame: frame["is_representative"]]
        .sort_values("weather_request_class")
        .reset_index(drop=True)
    )
    if len(representatives) != 28:
        raise ValueError("The class manifest must contain 28 representatives.")

    pending_count = int(manifest["run_status"].eq("pending").sum())
    print(f"Project root: {project_root}")
    print(f"Pending runs: {pending_count}")
    print(f"Complete runs: {manifest['run_status'].eq('complete').sum()}")
    print(
        f"Budgets: {args.hourly_runs} runs/hour and "
        f"{args.daily_runs} runs/24 hours"
    )
    if args.dry_run:
        print("Dry run: no requests were made.")
        return 0

    target = pending_count if args.continuous else min(args.max_runs, pending_count)
    pacer = RatePacer(
        rate_state_path,
        hourly_run_budget=args.hourly_runs,
        daily_run_budget=args.daily_runs,
    )
    session = requests.Session()
    completed_this_run = 0
    failed_this_run = 0
    consecutive_429 = 0
    start = time.monotonic()

    try:
        while completed_this_run < target:
            pending_indices = manifest.index[manifest["run_status"].eq("pending")]
            if len(pending_indices) == 0:
                break
            index = int(pending_indices[0])
            model_run = manifest.loc[index, "model_run_utc"]
            try:
                result = download_run(
                    manifest=manifest,
                    index=index,
                    project_root=project_root,
                    manifest_path=manifest_path,
                    representatives=representatives,
                    session=session,
                    pacer=pacer,
                    timeout_seconds=args.timeout_seconds,
                )
                completed_this_run += 1
                consecutive_429 = 0
                if (
                    completed_this_run % args.progress_every == 0
                    or completed_this_run == target
                ):
                    elapsed = (time.monotonic() - start) / 60
                    print(
                        f"Progress {completed_this_run}/{target}; last={model_run}; "
                        f"result={result}; elapsed={elapsed:.1f} min",
                        flush=True,
                    )
                if args.pause_seconds > 0:
                    time.sleep(args.pause_seconds)

            except RateLimitReached as error:
                consecutive_429 += 1
                if args.stop_on_429:
                    print(f"HTTP 429 at {model_run}; run remains pending.")
                    break
                if error.retry_after_seconds:
                    wait_seconds = error.retry_after_seconds + 10
                elif consecutive_429 == 1:
                    wait_seconds = 65 * 60
                else:
                    wait_seconds = seconds_until_next_utc_day()
                resume_at = datetime.fromtimestamp(
                    time.time() + wait_seconds, tz=timezone.utc
                )
                print(
                    f"HTTP 429 at {model_run}. Waiting {wait_seconds / 60:.1f} "
                    f"minutes; resume at {resume_at:%Y-%m-%d %H:%M:%S UTC}.",
                    flush=True,
                )
                time.sleep(wait_seconds)

            except Exception as error:
                failed_this_run += 1
                print(f"FAILED {model_run}: {type(error).__name__}: {error}", flush=True)

    except KeyboardInterrupt:
        print("Interrupted safely. Completed files and manifest progress were saved.")

    latest = pd.read_parquet(manifest_path)
    print("\nFinal manifest status:")
    print(latest["run_status"].value_counts(dropna=False))
    print(f"Completed/recovered this invocation: {completed_this_run}")
    print(f"Failed this invocation: {failed_this_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
