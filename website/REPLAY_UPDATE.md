# NYC Mobility Digital Twin July Replay

The replay displays every one of the 738 prepared July origins,
262 NYC zones and six forecast horizons: 1,160,136 forecast cases. EWR is excluded
from selection and correction; the underlying frozen model retains its original
263-zone training scope.

## Forecast modes

- Original XGBoost: nonnegative frozen-model forecasts.
- Full rolling correction (default): add the zone's mean available horizon-1
  residual over the last seven days.
- Damped correction: apply 25% of that adjustment.

The original forecast is retained; corrected predictions are clipped at zero.
The same correction applies to all six horizons. Corrections remain fixed at
issuance when the viewer advances the clock. Later actuals never revise an
already-issued forecast.

Counts become available two hours after their target hour ends. The correction
window is `(origin - 7 days, origin]` by publication time. Late-June horizon-1
forecasts initialize the replay with 165 hours per zone; coverage varies between
162 and 168 hours because the source has boundary gaps. The UI shows actual
history coverage and applies the same publication delay to revealed outcomes
and map activity.

The replay covers July 1 midnight through July 31 at 5 PM, NYC time. The last six
issuance hours are excluded because their full target windows extend into August.
The model artifact was created retrospectively; this is not an actual historical
deployment, live feed, or untouched confirmatory evaluation. Event as-of
availability remains unverified. Completed pickups do not measure unmet requests.

## Reproduce

From the project root, use the ride-demand environment and the prepared datasets and model:

```powershell
python scripts/predict_full_july.py --output-dir replay-output
```

The script retains the model hash, checks feature order/types, excludes EWR,
validates coverage and as-of eligibility, and writes a full-precision Parquet
artifact plus compact browser JSON. It refuses to overwrite an existing completed
browser artifact. The model is never retrained or overwritten. Browser values are
rounded to three decimals; full-precision predictions stay in the offline artifact.

Copy the generated `july-forecasts.json` to `website/dist/` to update forecasts.
`export_data.py` only rebuilds prepared counts/weather, not forecasts.
Run `node website/check-site.cjs` to verify forecast coverage, mode calculations,
publication timing, import rules, page labels and DOM references.

The six-hour CSV export includes original predictions, rolling correction,
selected correction strength and only already-revealed actuals. Imported CSV
predictions are used as supplied without another correction.

No raw trips, credentials or model weights are included in the browser deployment.
