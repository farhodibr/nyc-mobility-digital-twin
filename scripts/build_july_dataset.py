"""Prepare July replay inputs and separate outcomes; do not run any model."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Reuse the checked pilot's feature definitions without invoking its main().
PILOT_SCRIPT = Path(__file__).with_name('build_july_pilot.py')
if not PILOT_SCRIPT.exists():
    PILOT_SCRIPT = Path(__file__).with_name('build_july_simulation_pilot.py')
spec = importlib.util.spec_from_file_location('july_pilot_preparation', PILOT_SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
ROOT = pilot.ROOT
DEFAULT_OUTPUT = ROOT / 'data/gold/simulation/july_2026/dataset'
KEYS = ['forecast_origin_utc', 'forecast_horizon', 'location_id']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def weather_index():
    index = {}
    for path in (ROOT/'data/bronze/weather_forecasts').rglob('ecmwf_ifs_*.json'):
        match = re.search(r'(\d{8}T\d{4}Z)', path.name)
        if match:
            stamp = pd.to_datetime(match.group(1), format='%Y%m%dT%H%MZ', utc=True)
            index.setdefault(stamp, []).append(path)
    return index


def select_weather(origins, index):
    selections, documents = [], {}
    for origin in origins:
        cutoff = origin - pd.Timedelta(hours=7)
        expected = cutoff.floor('6h')
        for run in sorted((run for run in index if run <= cutoff), reverse=True):
            leads = [int((origin + pd.Timedelta(hours=h) - run)/pd.Timedelta(hours=1)) for h in range(1,7)]
            chosen = None
            for path in sorted(index[run], key=lambda p: ('forecast_hours=19' not in str(p), str(p))):
                if path not in documents:
                    documents[path] = json.loads(path.read_text(encoding='utf-8'))
                document = documents[path]
                if len(document.get('responses', [])) != 28 or len(document.get('representatives', [])) != 28:
                    continue
                if any(len(response['hourly']['time']) <= max(leads) for response in document['responses']):
                    continue
                assert pd.Timestamp(document['model_run_utc']) == run
                chosen = path
                break
            if chosen:
                selections.append({'origin': origin, 'run': run, 'expected_run': expected,
                                   'leads': leads, 'file': chosen})
                break
        else:
            raise ValueError(f'No usable eligible weather run for {origin}')
    assert len(selections) == len(origins)
    return selections, documents


def expand_weather(selections, documents, mapping):
    records = []
    manifest = []
    # Only ~124,000 request-class rows are constructed before zone expansion.
    for selection in selections:
        origin, run = selection['origin'], selection['run']
        document = documents[selection['file']]
        manifest.append({
            'forecast_origin_utc': origin,
            'forecast_origin_local': str(origin.tz_convert('America/New_York')),
            'selected_model_run_utc': run,
            'selected_run_available_utc': run + pd.Timedelta(hours=7),
            'is_fallback_weather_run': run != selection['expected_run'],
            'weather_file': str(selection['file']),
        })
        for representative, response in zip(document['representatives'], document['responses']):
            units = response['hourly_units']
            assert units['precipitation'] == units['snowfall'] == 'inch'
            assert units['wind_speed_10m'] in ['mp/h','mph'] and 'F' in units['temperature_2m']
            for horizon, lead in enumerate(selection['leads'], start=1):
                target = origin + pd.Timedelta(hours=horizon)
                valid = pd.to_datetime(response['hourly']['time'][lead], utc=True)
                assert target == valid
                row = {
                    'forecast_origin_utc': origin, 'forecast_horizon': horizon,
                    'target_hour_utc': target, 'selected_model_run_utc': run,
                    'selected_run_available_utc': run+pd.Timedelta(hours=7),
                    'selected_run_age_hours': int((origin-run)/pd.Timedelta(hours=1)),
                    'weather_lead_hours': lead,
                    'is_fallback_weather_run': run != selection['expected_run'],
                    'selected_forecast_hours': document['forecast_hours'],
                    'selected_weather_source': 'normal_archived_run' if run == selection['expected_run'] else 'fallback_archived_run',
                    'weather_request_class': representative['weather_request_class'],
                    'representative_location_id': representative['location_id'],
                    'request_latitude': representative['request_latitude'],
                    'request_longitude': representative['request_longitude'],
                    'model_grid_latitude': response['latitude'],
                    'model_grid_longitude': response['longitude'],
                    'response_elevation_m': response['elevation'], 'valid_time_utc': valid,
                }
                for raw, (value, flag) in pilot.WEATHER.items():
                    row[value] = response['hourly'][raw][lead]
                    row[flag] = row[value] is not None
                row['weather_forecast_complete'] = all(row[flag] for _,flag in pilot.WEATHER.values())
                row['weather_forecast_has_missing'] = not row['weather_forecast_complete']
                records.append(row)
    request_rows = pd.DataFrame(records)
    assert not request_rows.duplicated(['forecast_origin_utc','forecast_horizon','weather_request_class']).any()
    fields = ['weather_request_class','location_id','zone_latitude','zone_longitude',
              'is_coordinate_proxy','coordinate_source_location_id','class_zone_count']
    assert not mapping['location_id'].duplicated().any()
    expanded = request_rows.merge(mapping[fields], on='weather_request_class', validate='many_to_many')
    expanded['target_hour_local'] = expanded['target_hour_utc'].dt.tz_convert('America/New_York').dt.tz_localize(None)
    return expanded, pd.DataFrame(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is nonempty; no existing files will be replaced: {output}')
    output.mkdir(parents=True, exist_ok=True)
    artifact = pilot.ARTIFACT
    contract = json.loads((artifact/'feature_contract.json').read_text())
    metadata = json.loads((artifact/'metadata.json').read_text())
    model_path = artifact/'xgboost_model.json'
    initial_model_hash = sha(model_path)
    assert initial_model_hash == metadata['model_sha256']
    template = next((ROOT/'data/gold/production_model/hourly_zone_horizon_features').rglob('production_features_202606.parquet'))
    schema = pq.ParquetFile(template).schema_arrow.remove_metadata()
    template_hash = sha(template)
    pilot_dir = ROOT/'data/gold/simulation/july_2026/pilot'
    july_grid_path = pilot_dir/'zone_hour_pickups_202607.parquet'
    july_grid_hash = sha(july_grid_path)
    july = pd.read_parquet(july_grid_path)
    assert len(july) == 744*263
    assert not july.duplicated(['hour','location_id']).any()
    assert july['total_pickups'].notna().all() and july['total_pickups'].ge(0).all()
    assert july['hour'].min() == pd.Timestamp('2026-07-01 00:00')
    assert july['hour'].max() == pd.Timestamp('2026-07-31 23:00')
    definitions, functions = pilot.notebook_definitions()
    sports, concerts = pilot.event_keys(definitions, functions)
    context = pilot.add_context(july.rename(columns={'hour':'target_hour_local'}), sports, concerts,
                                definitions['unofficial_holidays_days'])

    # A historical overlap check catches changes to calendar/event definitions.
    reference = pq.ParquetFile(template).read(columns=['target_hour_local','location_id']+pilot.CONTEXT).to_pandas()
    reference = reference.drop_duplicates(['target_hour_local','location_id']).reset_index(drop=True)
    rebuilt = pilot.add_context(reference[['target_hour_local','location_id']],sports,concerts,
                                definitions['unofficial_holidays_days'])
    parity = {name:int((reference[name] != rebuilt[name]).sum()) for name in pilot.CONTEXT}
    assert not any(parity.values()), parity
    print(f'Calendar and event definitions agree on {len(reference):,} June target zone-hours.',flush=True)

    local_origins = pd.date_range('2026-07-01 00:00','2026-07-31 17:00',freq='h',tz='America/New_York')
    origins = local_origins.tz_convert('UTC')
    assert len(origins) == 738
    mapping_file = next((ROOT/'data/silver/weather_forecasts').rglob('nyc_weather_request_class_manifest.parquet'))
    mapping = pd.read_parquet(mapping_file)
    assert set(mapping['location_id']) == set(range(1,264))
    selections, documents = select_weather(origins,weather_index())
    print(f'Selected weather for {len(origins)} origins from {len(set(s["file"] for s in selections))} archived files.',flush=True)
    frame, origin_manifest = expand_weather(selections,documents,mapping)
    assert len(frame) == 738*263*6 == 1164564
    assert not frame.duplicated(KEYS).any()
    assert frame.groupby('forecast_origin_utc').size().eq(1578).all()
    assert frame.groupby(['forecast_origin_utc','forecast_horizon']).size().eq(263).all()
    assert frame['selected_run_available_utc'].le(frame['forecast_origin_utc']).all()
    assert frame['target_hour_local'].ge(pd.Timestamp('2026-07-01')).all()
    assert frame['target_hour_local'].lt(pd.Timestamp('2026-08-01')).all()
    print('Weather expanded to 1,164,564 origin–horizon–zone rows.',flush=True)

    historical_source = ROOT/'data/gold/zone_hour_pickups_weather_E_with_concerts.parquet'
    history = pd.read_parquet(historical_source, columns=['hour','location_id','total_pickups'],
                              filters=[('hour','>=',pd.Timestamp('2026-06-15')),('hour','<',pd.Timestamp('2026-07-01'))])
    history = pd.concat([history,july],ignore_index=True)
    assert not history.duplicated(['hour','location_id']).any()
    lookup = history.set_index(['hour','location_id'])['total_pickups']
    lag_source_checks = {}
    for hours in [24,168,336]:
        source = frame['target_hour_local'] - pd.Timedelta(hours=hours)
        source_utc = source.dt.tz_localize('America/New_York').dt.tz_convert('UTC')
        assert source_utc.lt(frame['forecast_origin_utc']).all()
        keys = pd.MultiIndex.from_arrays([source,frame['location_id']])
        frame[f'pickup_lag_{hours}'] = lookup.reindex(keys).to_numpy()
        assert frame[f'pickup_lag_{hours}'].notna().all()
        lag_source_checks[str(hours)] = {'first_source_local':str(source.min()),
                                        'last_source_local':str(source.max()),'all_before_origin':True}
    frame['is_ambiguous_dst_target'] = False
    frame = frame.merge(context.rename(columns={'total_pickups':'_actual_pickups'}),
                         on=['target_hour_local','location_id'],how='left',validate='many_to_one')
    frame = frame.rename(columns={'_actual_pickups':'total_pickups'})
    assert set(frame.columns) == set(schema.names)
    assert frame['total_pickups'].notna().all()
    weather_columns = [value for value,_ in pilot.WEATHER.values()]
    for column in frame.columns:
        if column not in weather_columns:
            assert frame[column].notna().all(),column
    for value,flag in pilot.WEATHER.values():
        assert frame[flag].eq(frame[value].notna()).all()
    assert frame['weather_forecast_complete'].eq(frame[weather_columns].notna().all(axis=1)).all()
    frame = frame[schema.names].sort_values(KEYS).reset_index(drop=True)
    table = pa.Table.from_pandas(frame,schema=schema,preserve_index=False,safe=True)
    assert table.schema.equals(schema)
    typed = table.to_pandas()
    first = typed.loc[typed['forecast_origin_utc']==origins[0]].reset_index(drop=True)
    expected_first = pd.read_parquet(pilot_dir/'pilot_features_20260701_0000_NY.parquet')
    pd.testing.assert_frame_equal(first,expected_first,check_exact=True)
    inputs = pilot.model_matrix(typed,contract)
    assert inputs.shape == (1164564,30)
    assert 'total_pickups' not in inputs
    # Dtypes are checked without constructing a model or requesting predictions.
    expected_types = []
    for column in inputs:
        dtype = inputs[column].dtype
        if isinstance(dtype,pd.CategoricalDtype):
            expected_types.append('c')
        elif pd.api.types.is_bool_dtype(dtype):
            expected_types.append('i')
        elif pd.api.types.is_float_dtype(dtype):
            expected_types.append('float')
        elif pd.api.types.is_integer_dtype(dtype):
            expected_types.append('int')
        else:
            raise TypeError(f'Unexpected model input dtype: {column} {dtype}')
    assert expected_types == contract['xgboost_feature_types']

    full_file = output/'production_features_202607_simulation.parquet'
    input_file = output/'model_inputs_202607.parquet'
    outcome_file = output/'scoring_outcomes_202607.parquet'
    pq.write_table(table,full_file,compression='zstd')
    inputs.to_parquet(input_file,index=False,compression='zstd')
    outcomes = typed[['forecast_origin_utc','forecast_horizon','target_hour_utc','target_hour_local','location_id','total_pickups']]
    outcomes.to_parquet(outcome_file,index=False,compression='zstd')
    origin_manifest.to_parquet(output/'forecast_origin_manifest_202607.parquet',index=False,compression='zstd')
    origin_manifest.to_csv(output/'forecast_origin_manifest_202607.csv',index=False)
    reloaded = pilot.model_matrix(pd.read_parquet(input_file),contract)
    pd.testing.assert_frame_equal(inputs,reloaded,check_exact=True)
    readback = pq.ParquetFile(full_file)
    assert readback.metadata.num_rows == 1164564
    assert readback.schema_arrow.remove_metadata().equals(schema)
    actual_check = pd.read_parquet(outcome_file)
    pd.testing.assert_frame_equal(outcomes.reset_index(drop=True),actual_check,check_exact=True)
    assert sha(model_path) == initial_model_hash
    assert sha(template) == template_hash
    assert sha(july_grid_path) == july_grid_hash

    category_contract = json.loads((pilot_dir/'pilot_input_contract.json').read_text())
    (output/'model_input_contract.json').write_text(json.dumps(category_contract,indent=2)+'\n')
    selected_files = sorted(set(s['file'] for s in selections))
    provenance = {'source_notebook':str(ROOT/'notebooks/data_engineering1_organized_v3.ipynb'),
                  'source_notebook_sha256':sha(ROOT/'notebooks/data_engineering1_organized_v3.ipynb'),
                  'builder_sha256':sha(Path(__file__)),
                  'pilot_builder_sha256':sha(PILOT_SCRIPT),
                  'training_schema_reference':str(template),'training_reference_sha256':template_hash,
                  'july_hourly_grid':str(july_grid_path),'july_hourly_grid_sha256':july_grid_hash,
                  'saved_model':str(model_path),'saved_model_sha256':initial_model_hash,
                  'weather_files':[{'path':str(p),'sha256':sha(p)} for p in selected_files]}
    (output/'source_manifest.json').write_text(json.dumps(provenance,indent=2)+'\n')
    unique_targets = typed.drop_duplicates(['target_hour_local','location_id'])
    validation = {
        'status':'july_dataset_prepared_and_validated_no_predictions',
        'rows':len(typed),'origins':len(origins),'zones':263,'horizons':[1,2,3,4,5,6],
        'columns':len(schema),'model_features':len(inputs.columns),
        'first_origin_local':str(local_origins[0]),'last_origin_local':str(local_origins[-1]),
        'first_target_local':str(typed['target_hour_local'].min()),
        'last_target_local':str(typed['target_hour_local'].max()),
        'excluded_late_july_origins':6,
        'scope':'738 New York local July origins with all six targets inside July; 263-zone evaluation scope including EWR',
        'schema_matches_training':True,'input_order_and_types_match_saved_contract':True,
        'category_encoding_round_trip_passed':True,'first_origin_exactly_matches_checked_pilot':True,
        'calendar_event_reference_rows':len(reference),'calendar_event_mismatches':parity,
        'missing_target_rows':0,'missing_lag_rows':0,'duplicate_keys':0,
        'weather_files_used':len(selected_files),
        'fallback_weather_origins':int(origin_manifest['is_fallback_weather_run'].sum()),
        'incomplete_weather_rows':int(typed['weather_forecast_has_missing'].sum()),
        'weather_missing_values_by_column':{name:int(typed[name].isna().sum()) for name in weather_columns},
        'weather_availability_policy_passed':True,'lag_source_checks':lag_source_checks,
        'distinct_target_zone_hours':len(unique_targets),
        'event_distinct_target_zone_hours':{name:int(unique_targets[name].sum()) for name in ['is_post_game_demand','is_concert_window','is_wide_concert_window']},
        'model_retrained':False,'predictions_generated':False,'accuracy_metrics_calculated':False,
        'training_reference_and_model_unchanged':True,
        'assumptions':json.loads((pilot_dir/'pilot_validation.json').read_text())['assumptions'],
        'output_files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in [full_file,input_file,outcome_file]},
    }
    (output/'dataset_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    preview = typed.loc[typed['location_id'].isin([93,132,138,186]) & typed['forecast_origin_utc'].isin([origins[0],origins[72],origins[-1]]),
                       ['forecast_origin_utc','forecast_horizon','target_hour_local','location_id','pickup_lag_24','pickup_lag_168','pickup_lag_336','forecast_temperature_f','is_holiday','is_post_game_demand','is_concert_window','is_wide_concert_window']]
    preview.to_csv(output/'dataset_preview.csv',index=False)
    print(json.dumps({k:v for k,v in validation.items() if k not in ['calendar_event_mismatches','assumptions','output_files','lag_source_checks']},indent=2),flush=True)


if __name__=='__main__':
    main()
