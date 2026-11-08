"""Build one July replay origin using the frozen evaluation dataset contract.

Only this simulation output directory is written. Existing data/model artifacts
are read-only. The script does not fit a model or generate performance scores.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

import duckdb
import holidays
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'models/evaluation/xgboost_263_zone_evaluation_2026-08-25'
CONTEXT = ['hour_of_day', 'day_of_week', 'is_weekend', 'month',
           'is_weekday_rush_hour', 'is_weekend_night', 'is_holiday',
           'is_unofficial_holiday', 'is_post_game_demand',
           'is_concert_window', 'is_wide_concert_window']
WEATHER = {'temperature_2m': ('forecast_temperature_f', 'temperature_forecast_available'),
           'precipitation': ('forecast_precipitation_in', 'precipitation_forecast_available'),
           'rain': ('forecast_rain_in', 'rain_forecast_available'),
           'snowfall': ('forecast_snowfall_in', 'snowfall_forecast_available'),
           'wind_speed_10m': ('forecast_wind_speed_mph', 'wind_forecast_available')}


def notebook_definitions():
    """Reuse literal mappings and pure functions without running notebook cells."""
    p = ROOT / 'notebooks/data_engineering1_organized_v3.ipynb'
    n = json.loads(p.read_text(encoding='utf-8'))
    literals = {}
    namespace = {'pd': pd}
    for cell in n['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            nodes = ast.parse(''.join(cell.get('source', []))).body
        except SyntaxError:
            continue  # Notebook magics do not belong to the definitions needed here.
        for node in nodes:
            if isinstance(node, ast.Assign):
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id in {
                        'team_zone_map', 'concert_venues_final', 'unofficial_holidays_days'
                    }:
                        literals[name.id] = ast.literal_eval(node.value)
            if isinstance(node, ast.FunctionDef) and node.name in {
                'estimate_game_duration', 'get_affected_hours'
            }:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), 'exec'), namespace)
    assert set(literals) == {'team_zone_map', 'concert_venues_final', 'unofficial_holidays_days'}
    return literals, namespace


def event_keys(definitions, functions):
    sports = set()
    for name, info in definitions['team_zone_map'].items():
        safe = name.lower().replace(' ', '_')
        for file in sorted((ROOT / 'data/external/events').glob(f'{safe}_*_schedule.json')):
            doc = json.loads(file.read_text(encoding='utf-8'))
            for event in doc['events']:
                competitors = event['competitions'][0]['competitors']
                home = next(c for c in competitors if c['homeAway'] == 'home')
                if str(home['id']) != str(info['team_id']):
                    continue
                start = pd.Timestamp(event['date']).tz_convert('America/New_York')
                if not pd.Timestamp('2026-05-01', tz='America/New_York') <= start < pd.Timestamp('2026-08-02', tz='America/New_York'):
                    continue
                end = start + functions['estimate_game_duration'](info['sport'], start)
                for hour in functions['get_affected_hours'](end, end + pd.Timedelta(hours=1)):
                    sports.add((hour, int(info['location_id'])))

    concerts = set()
    for venue in definitions['concert_venues_final']:
        safe = venue['venue_name'].lower().replace(' ', '_')
        file = ROOT / f'data/external/events/concerts/{safe}_2026_setlists.json'
        doc = json.loads(file.read_text(encoding='utf-8'))
        for event in doc['setlist']:
            date = pd.to_datetime(event['eventDate'], format='%d-%m-%Y').date()
            concerts.add((date, int(venue['location_id'])))
    return sports, concerts


def add_context(frame, sports, concerts, unofficial):
    frame = frame.copy()
    time = pd.to_datetime(frame['target_hour_local'])
    frame['hour_of_day'] = time.dt.hour
    frame['day_of_week'] = time.dt.dayofweek
    frame['month'] = time.dt.month
    frame['is_weekend'] = frame['day_of_week'].isin([5, 6])
    frame['is_weekday_rush_hour'] = ~frame['is_weekend'] & (
        frame['hour_of_day'].between(7, 9) | frame['hour_of_day'].between(16, 18))
    frame['is_weekend_night'] = frame['is_weekend'] & (
        (frame['hour_of_day'] >= 22) | (frame['hour_of_day'] <= 2))
    calendar = holidays.US(state='NY', years=[2026])
    frame['is_holiday'] = [date in calendar for date in time.dt.date]
    frame['is_unofficial_holiday'] = [(date.month, date.day) in unofficial for date in time.dt.date]
    keys = list(zip(time, frame['location_id'].astype(int)))
    frame['is_post_game_demand'] = [key in sports for key in keys]
    dates = list(zip(time.dt.date, frame['location_id'].astype(int)))
    previous = list(zip((time - pd.Timedelta(days=1)).dt.date, frame['location_id'].astype(int)))
    same_day = np.array([key in concerts for key in dates])
    prior_day = np.array([key in concerts for key in previous])
    frame['is_concert_window'] = same_day & frame['hour_of_day'].between(22, 23)
    frame['is_wide_concert_window'] = frame['is_concert_window'] | (prior_day & (frame['hour_of_day'] == 0))
    return frame


def weather_rows(origin):
    archive = ROOT / 'data/bronze/weather_forecasts'
    index = {}
    for p in archive.rglob('ecmwf_ifs_*.json'):
        match = re.search(r'(\d{8}T\d{4}Z)', p.name)
        if match:
            stamp = pd.to_datetime(match.group(1), format='%Y%m%dT%H%MZ', utc=True)
            index.setdefault(stamp, []).append(p)
    cutoff = origin - pd.Timedelta(hours=7)
    expected = cutoff.floor('6h')
    eligible = sorted((t for t in index if t <= cutoff), reverse=True)
    mapping_file = next((ROOT / 'data/silver/weather_forecasts').rglob('nyc_weather_request_class_manifest.parquet'))
    mapping = pd.read_parquet(mapping_file)
    assert mapping['location_id'].nunique() == 263
    zone_columns = ['weather_request_class', 'location_id', 'zone_latitude', 'zone_longitude',
                    'is_coordinate_proxy', 'coordinate_source_location_id', 'class_zone_count']
    for run in eligible:
        leads = [int((origin + pd.Timedelta(hours=h) - run) / pd.Timedelta(hours=1)) for h in range(1, 7)]
        paths = sorted(index[run], key=lambda p: ('forecast_hours=19' not in str(p), str(p)))
        for path in paths:
            doc = json.loads(path.read_text(encoding='utf-8'))
            if len(doc.get('responses', [])) != 28 or len(doc.get('representatives', [])) != 28:
                continue
            if any(len(response['hourly']['time']) <= max(leads) for response in doc['responses']):
                continue
            assert pd.Timestamp(doc['model_run_utc']) == run
            records = []
            for representative, response in zip(doc['representatives'], doc['responses']):
                units = response['hourly_units']
                assert units['precipitation'] == 'inch' and units['snowfall'] == 'inch'
                assert units['wind_speed_10m'] in ['mp/h', 'mph']
                assert 'F' in units['temperature_2m']
                for h, lead in enumerate(leads, start=1):
                    target = origin + pd.Timedelta(hours=h)
                    valid = pd.to_datetime(response['hourly']['time'][lead], utc=True)
                    assert valid == target
                    row = dict(forecast_origin_utc=origin, forecast_horizon=h, target_hour_utc=target,
                               selected_model_run_utc=run, selected_run_available_utc=run + pd.Timedelta(hours=7),
                               selected_run_age_hours=int((origin-run)/pd.Timedelta(hours=1)),
                               weather_lead_hours=lead, is_fallback_weather_run=run != expected,
                               selected_forecast_hours=doc['forecast_hours'],
                               selected_weather_source='normal_archived_run' if run == expected else 'fallback_archived_run',
                               weather_request_class=representative['weather_request_class'],
                               representative_location_id=representative['location_id'],
                               request_latitude=representative['request_latitude'],
                               request_longitude=representative['request_longitude'],
                               model_grid_latitude=response['latitude'], model_grid_longitude=response['longitude'],
                               response_elevation_m=response['elevation'], valid_time_utc=valid)
                    for raw, (value, flag) in WEATHER.items():
                        row[value] = response['hourly'][raw][lead]
                        row[flag] = row[value] is not None
                    row['weather_forecast_complete'] = all(row[flag] for _, flag in WEATHER.values())
                    row['weather_forecast_has_missing'] = not row['weather_forecast_complete']
                    records.append(row)
            result = pd.DataFrame(records).merge(mapping[zone_columns], on='weather_request_class', validate='many_to_many')
            result['target_hour_local'] = result['target_hour_utc'].dt.tz_convert('America/New_York').dt.tz_localize(None)
            assert len(result) == 1578
            assert result['selected_run_available_utc'].le(origin).all()
            return result, path
    raise ValueError('No usable archived weather run satisfies the seven-hour policy.')


def model_matrix(frame, contract):
    result = frame[contract['ordered_feature_names']].copy()
    categories = {'location_id': range(1, 264), 'forecast_horizon': range(1, 7),
                  'hour_of_day': range(24), 'day_of_week': range(7), 'month': range(1, 13)}
    for name, levels in categories.items():
        result[name] = pd.Categorical(result[name], categories=list(levels))
        assert not result[name].isna().any()
    assert 'total_pickups' not in result
    assert list(result.columns) == contract['ordered_feature_names']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT/'data/gold/simulation/july_2026/pilot')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Nonempty output directory retained without changes: {output}')
    output.mkdir(parents=True, exist_ok=True)
    contract = json.loads((ARTIFACT/'feature_contract.json').read_text())
    model = ARTIFACT/'xgboost_model.json'
    model_hash = hashlib.sha256(model.read_bytes()).hexdigest()
    assert model_hash == json.loads((ARTIFACT/'metadata.json').read_text())['model_sha256']
    template = next((ROOT/'data/gold/production_model/hourly_zone_horizon_features').rglob('production_features_202606.parquet'))
    schema = pq.ParquetFile(template).schema_arrow.remove_metadata()
    definitions, functions = notebook_definitions()
    sports, concerts = event_keys(definitions, functions)

    # Recreate every calendar/event flag on June targets before extrapolating to July.
    reference = pq.ParquetFile(template).read(columns=['target_hour_local','location_id'] + CONTEXT).to_pandas()
    reference = reference.drop_duplicates(['target_hour_local','location_id']).sort_values(['target_hour_local','location_id']).reset_index(drop=True)
    rebuilt = add_context(reference[['target_hour_local','location_id']], sports, concerts, definitions['unofficial_holidays_days'])
    mismatches = {name: int((reference[name] != rebuilt[name]).sum()) for name in CONTEXT}
    if any(mismatches.values()):
        (output/'context_mismatches.json').write_text(json.dumps(mismatches, indent=2))
        raise ValueError(f'Rebuilt context differs from training preparation: {mismatches}')
    print(f'Calendar/event parity passed on {len(reference):,} unique June target zone-hours.', flush=True)

    raw = ROOT/'data/bronze/fhvhv_tripdata_2026-07.parquet'
    con = duckdb.connect(':memory:')
    con.execute("SET memory_limit='1GB'")
    july_counts = con.execute('''
        SELECT date_trunc('hour', pickup_datetime) AS hour,
               CAST(PULocationID AS SMALLINT) AS location_id, count(*) AS total_pickups
        FROM read_parquet(?)
        WHERE pickup_datetime >= TIMESTAMP '2026-07-01'
          AND pickup_datetime < TIMESTAMP '2026-08-01'
          AND PULocationID BETWEEN 1 AND 263
        GROUP BY 1, 2
    ''', [str(raw)]).fetchdf()
    grid = pd.MultiIndex.from_product([pd.date_range('2026-07-01','2026-08-01',freq='h',inclusive='left'), range(1,264)], names=['hour','location_id']).to_frame(index=False)
    july = grid.merge(july_counts, on=['hour','location_id'], how='left', validate='one_to_one')
    july['total_pickups'] = july['total_pickups'].fillna(0).astype('int32')
    assert len(july) == 195672
    assert int(july['total_pickups'].sum()) == int(july_counts['total_pickups'].sum())
    history = con.execute('''
        SELECT hour, location_id, total_pickups FROM read_parquet(?)
        WHERE hour >= TIMESTAMP '2026-05-15' AND hour < TIMESTAMP '2026-07-01'
    ''', [str(ROOT/'data/gold/zone_hour_pickups_weather_E_with_concerts.parquet')]).fetchdf()
    con.close()
    history = pd.concat([history,july], ignore_index=True)
    assert not history.duplicated(['hour','location_id']).any()
    lookup = history.set_index(['hour','location_id'])['total_pickups']
    origin = pd.Timestamp('2026-07-01 00:00', tz='America/New_York').tz_convert('UTC')
    frame, weather_file = weather_rows(origin)
    targets = pd.MultiIndex.from_arrays([frame['target_hour_local'],frame['location_id']])
    frame['total_pickups'] = lookup.reindex(targets).to_numpy()
    assert frame['total_pickups'].notna().all()
    for hours in [24,168,336]:
        source = frame['target_hour_local'] - pd.Timedelta(hours=hours)
        assert source.dt.tz_localize('America/New_York').dt.tz_convert('UTC').lt(origin).all()
        lag_keys = pd.MultiIndex.from_arrays([source,frame['location_id']])
        frame[f'pickup_lag_{hours}'] = lookup.reindex(lag_keys).to_numpy()
        assert frame[f'pickup_lag_{hours}'].notna().all()
    frame['is_ambiguous_dst_target'] = False
    frame = add_context(frame, sports, concerts, definitions['unofficial_holidays_days'])
    assert set(frame.columns) == set(schema.names)
    frame = frame[schema.names].sort_values(['forecast_origin_utc','forecast_horizon','location_id']).reset_index(drop=True)
    assert not frame.duplicated(['forecast_origin_utc','forecast_horizon','location_id']).any()
    for field in schema:
        if field.name not in [value for value, _ in WEATHER.values()]:
            assert not frame[field.name].isna().any(), field.name
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    assert table.schema.equals(schema)
    matrix = model_matrix(table.to_pandas(), contract)
    assert matrix.shape == (1578,30)
    dmatrix = xgb.DMatrix(matrix, enable_categorical=True)
    saved_model = xgb.Booster()
    saved_model.load_model(model)
    assert dmatrix.feature_names == saved_model.feature_names == contract['ordered_feature_names']
    assert dmatrix.feature_types == saved_model.feature_types == contract['xgboost_feature_types']
    july.to_parquet(output/'zone_hour_pickups_202607.parquet', index=False, compression='zstd')
    pq.write_table(table, output/'pilot_features_20260701_0000_NY.parquet', compression='zstd')
    matrix.to_parquet(output/'pilot_model_inputs_20260701_0000_NY.parquet', index=False, compression='zstd')
    # Numeric categorical levels are not reliably retained as pandas categories
    # through Parquet. Apply the same explicit encoding used after training reads.
    reloaded_matrix = model_matrix(pd.read_parquet(output/'pilot_model_inputs_20260701_0000_NY.parquet'), contract)
    pd.testing.assert_frame_equal(matrix, reloaded_matrix)
    reloaded_dmatrix = xgb.DMatrix(reloaded_matrix, enable_categorical=True)
    assert reloaded_dmatrix.feature_names == saved_model.feature_names
    assert reloaded_dmatrix.feature_types == saved_model.feature_types
    input_contract = dict(contract)
    input_contract['categorical_levels'] = {'location_id': list(range(1,264)), 'forecast_horizon': list(range(1,7)),
                                             'hour_of_day': list(range(24)), 'day_of_week': list(range(7)), 'month': list(range(1,13))}
    input_contract['load_instruction'] = 'Select ordered_feature_names, then reapply the categorical_levels with pd.Categorical before constructing the XGBoost matrix.'
    (output/'pilot_input_contract.json').write_text(json.dumps(input_contract,indent=2)+'\n')
    scoring = frame[['forecast_origin_utc','forecast_horizon','target_hour_utc','target_hour_local','location_id','total_pickups']].copy()
    scoring.to_parquet(output/'pilot_scoring_outcomes_20260701_0000_NY.parquet', index=False, compression='zstd')
    preview_columns = ['forecast_origin_utc','forecast_horizon','target_hour_local','location_id','pickup_lag_24','pickup_lag_168','pickup_lag_336','forecast_temperature_f','is_post_game_demand','is_concert_window','is_wide_concert_window']
    frame.loc[frame['location_id'].isin([93,132,138,186]),preview_columns].to_csv(output/'pilot_preview.csv',index=False)
    report = {
        'status': 'single_origin_dataset_validated',
        'forecast_origin_local': str(origin.tz_convert('America/New_York')),
        'forecast_origin_utc': str(origin),
        'target_hours_local': sorted(str(t) for t in frame['target_hour_local'].unique()),
        'rows': len(frame), 'zones': frame['location_id'].nunique(), 'horizons': 6,
        'dataset_columns': len(schema), 'model_features': matrix.shape[1],
        'schema_matches_training': True, 'categorical_levels_match_training': True,
        'xgboost_input_names_and_types_match_saved_model': True,
        'parquet_round_trip_with_explicit_category_encoding_passed': True,
        'context_reference_rows_checked': len(reference), 'context_mismatches': mismatches,
        'selected_weather_file': str(weather_file),
        'selected_model_run_utc': str(frame['selected_model_run_utc'].iloc[0]),
        'selected_run_available_utc': str(frame['selected_run_available_utc'].iloc[0]),
        'weather_selected_under_seven_hour_policy': True,
        'lag_source_hours_all_before_origin': True,
        'july_hourly_grid_rows': len(july), 'july_supported_zone_pickups': int(july['total_pickups'].sum()),
        'july_zero_pickup_zone_hours': int((july['total_pickups']==0).sum()),
        'model_path': str(model), 'model_sha256': model_hash,
        'model_retrained': False, 'predictions_generated': False,
        'full_july_forecasting_dataset_built': False,
        'assumptions': [
            'Historical replay assumes observed lag counts are available by issuance; real-time TLC delivery is not claimed.',
            'Seven-hour weather eligibility reproduces the training policy, not independently verified original publication timing.',
            'Retrospective event files and existing venue mappings are reused to match training; historical as-of event availability is not established.',
            'The saved evaluation model has 263 zones including EWR; no geographic retraining or scope change is made.',
        ],
    }
    assert hashlib.sha256(model.read_bytes()).hexdigest() == model_hash
    (output/'pilot_validation.json').write_text(json.dumps(report,indent=2,default=lambda v: int(v))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['context_mismatches','assumptions']},indent=2,default=lambda v:int(v)),flush=True)


if __name__ == '__main__':
    main()
