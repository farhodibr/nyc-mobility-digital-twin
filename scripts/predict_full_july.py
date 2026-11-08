"""Frozen-model July replay with as-of seven-day residual corrections.

Writes new experiment artifacts only; does not retrain or modify the model.
"""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'july-forecasts.json').exists():
        raise FileExistsError('Existing replay artifacts retained.')
    root = args.project_root.resolve()
    artifact = root / 'models/evaluation/xgboost_263_zone_evaluation_2026-08-25'
    contract = json.loads((root / 'data/gold/simulation/july_2026/dataset/model_input_contract.json').read_text())
    features = contract['ordered_feature_names']
    model_path = artifact / 'xgboost_model.json'
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = json.loads((artifact / 'metadata.json').read_text())
    assert model_hash == metadata['model_sha256']
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.set_param({'device': 'cpu'})
    keys = ['forecast_origin_utc', 'target_hour_utc', 'forecast_horizon', 'location_id', 'total_pickups']
    columns = list(dict.fromkeys(features + keys))
    july_path = root / 'data/gold/simulation/july_2026/dataset/production_features_202607_simulation.parquet'
    july = pq.ParquetFile(july_path).read(columns=columns).to_pandas()
    june_path = root / 'data/gold/production_model/hourly_zone_horizon_features/availability_delay=7h/year=2026/month=06/production_features_202606.parquet'
    june = pq.ParquetFile(june_path).read(columns=columns).to_pandas()
    for frame in (july, june):
        for column in ('forecast_origin_utc', 'target_hour_utc'):
            frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit('ns')
    first_origin = july.forecast_origin_utc.min()
    june = june.loc[(june.forecast_horizon == 1) & (june.location_id != 1)
                    & (june.target_hour_utc + pd.Timedelta(hours=3) > first_origin - pd.Timedelta(days=7))
                    & (june.target_hour_utc + pd.Timedelta(hours=3) <= first_origin)].copy()
    july = july.loc[july.location_id != 1].copy()
    origins = pd.DatetimeIndex(sorted(july.forecast_origin_utc.unique())).as_unit('ns')
    assert len(origins) == 738 and july.shape[0] == 738 * 262 * 6
    assert not july.duplicated(['forecast_origin_utc', 'location_id', 'forecast_horizon']).any()

    def predict(frame):
        matrix_data = frame[features].copy()
        for name, levels in contract['categorical_levels'].items():
            matrix_data[name] = pd.Categorical(matrix_data[name], categories=levels)
            assert matrix_data[name].notna().all()
        matrix = xgb.DMatrix(matrix_data, enable_categorical=True)
        assert matrix.feature_names == booster.feature_names
        assert matrix.feature_types == booster.feature_types
        prediction = np.maximum(booster.predict(matrix), 0).astype('float64')
        assert np.isfinite(prediction).all()
        return prediction

    print('Generating late-June horizon-1 forecasts...', flush=True)
    june['original_prediction'] = predict(june)
    print('Generating all July forecasts...', flush=True)
    july['original_prediction'] = predict(july)
    history = pd.concat([june, july.loc[july.forecast_horizon == 1]], ignore_index=True)
    history['available_at'] = history.target_hour_utc + pd.Timedelta(hours=3)
    history['residual'] = history.total_pickups - history.original_prediction
    assert not history.duplicated(['location_id', 'target_hour_utc']).any()
    zone_payload = []
    correction_frames = []
    for zone, frame in history.groupby('location_id'):
        frame = frame.sort_values('available_at')
        times = frame.available_at.astype('int64').to_numpy()
        query = origins.astype('int64').to_numpy()
        left = np.searchsorted(times, query - pd.Timedelta(days=7).value, side='right')
        right = np.searchsorted(times, query, side='right')
        counts = right - left
        assert (counts > 0).all()
        cumulative = np.r_[0., frame.residual.cumsum().to_numpy()]
        correction = (cumulative[right] - cumulative[left]) / counts
        # Every incorporated outcome is published no later than issuance.
        assert (times[right - 1] <= query).all()
        baseline = july.loc[july.location_id == zone].pivot(index='forecast_origin_utc', columns='forecast_horizon', values='original_prediction').reindex(origins)
        assert baseline.shape == (738, 6) and baseline.notna().all().all()
        zone_payload.append({'id': int(zone), 'original': np.round(baseline.to_numpy().ravel(), 3).tolist(),
                             'correction': np.round(correction, 3).tolist(), 'historyCount': counts.tolist()})
        correction_frames.append(pd.DataFrame({'forecast_origin_utc': origins, 'location_id': int(zone),
                                               'rolling_correction': correction, 'history_count': counts}))
    result = july[keys + ['original_prediction']].merge(pd.concat(correction_frames), on=['forecast_origin_utc', 'location_id'], validate='many_to_one')
    result['full_prediction'] = np.maximum(0, result.original_prediction + result.rolling_correction)
    result['damped_prediction'] = np.maximum(0, result.original_prediction + .25 * result.rolling_correction)
    result.to_parquet(out / 'july_replay_predictions.parquet', index=False, compression='zstd')
    manifest = {'artifactId': metadata['artifact_id'], 'modelSha256': model_hash, 'originCount': 738,
                'cases': len(result), 'zoneCount': 262, 'horizons': 6, 'firstOriginUtc': first_origin.isoformat(),
                'publicationDelayAfterHourEnd': 2, 'windowHours': 168, 'firstHistoryHours': int(zone_payload[0]['historyCount'][0]),
                'mode': 'historical replay; model artifact created retrospectively', 'modelRetrained': False,
                'originalModelScope': 263, 'displayScope': 262,
                'julyInputSha256': hashlib.sha256(july_path.read_bytes()).hexdigest()}
    (out / 'prediction_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (out / 'july-forecasts.json').write_text(json.dumps({**manifest, 'zones': zone_payload}, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_hash
    print(json.dumps(manifest, indent=2), flush=True)

if __name__ == '__main__':
    main()
