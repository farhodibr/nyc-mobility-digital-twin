"""Run frozen-model inference for the checked July origin only; do not score."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT/'models/evaluation/xgboost_263_zone_evaluation_2026-08-25'
PILOT = ROOT/'data/gold/simulation/july_2026/pilot'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'data/gold/simulation/july_2026/predictions/first_origin')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Existing prediction output retained: {output}')
    output.mkdir(parents=True,exist_ok=True)
    contract = json.loads((PILOT/'pilot_input_contract.json').read_text())
    metadata = json.loads((ARTIFACT/'metadata.json').read_text())
    model_path = ARTIFACT/'xgboost_model.json'
    before = sha(model_path)
    assert before == metadata['model_sha256']
    input_file = PILOT/'pilot_model_inputs_20260701_0000_NY.parquet'
    features = pd.read_parquet(input_file)
    assert list(features.columns) == contract['ordered_feature_names']
    assert features.shape == (1578,30)
    assert 'total_pickups' not in features
    for name,levels in contract['categorical_levels'].items():
        features[name] = pd.Categorical(features[name],categories=levels)
        assert not features[name].isna().any()
    keys = pq.ParquetFile(PILOT/'pilot_features_20260701_0000_NY.parquet').read(columns=[
        'forecast_origin_utc','forecast_horizon','target_hour_utc','target_hour_local','location_id']).to_pandas()
    assert len(keys) == len(features)
    assert keys['forecast_origin_utc'].nunique() == 1
    assert keys['forecast_origin_utc'].iloc[0] == pd.Timestamp('2026-07-01T04:00Z')
    assert keys['location_id'].eq(features['location_id'].astype(int)).all()
    assert keys['forecast_horizon'].eq(features['forecast_horizon'].astype(int)).all()
    assert not keys.duplicated(['forecast_origin_utc','forecast_horizon','location_id']).any()
    booster = xgb.Booster()
    booster.load_model(model_path)
    booster.set_param({'device':'cpu'})
    matrix = xgb.DMatrix(features,enable_categorical=True)
    assert matrix.feature_names == booster.feature_names == contract['ordered_feature_names']
    assert matrix.feature_types == booster.feature_types == contract['xgboost_feature_types']
    rounds = booster.num_boosted_rounds()
    assert rounds == metadata['training_contract']['boosting_rounds'] == 1613
    raw = booster.predict(matrix,iteration_range=(0,rounds),validate_features=True)
    assert raw.shape == (1578,) and np.isfinite(raw).all()
    predictions = np.maximum(raw,0)
    result = keys.copy()
    result['prediction'] = predictions
    assert result.groupby('location_id').size().eq(6).all()
    assert result.groupby('forecast_horizon').size().eq(263).all()
    assert sha(model_path) == before
    result.to_parquet(output/'predictions_20260701_0000_NY.parquet',index=False,compression='zstd')
    result.to_csv(output/'predictions_20260701_0000_NY.csv',index=False)
    generated = datetime.now(timezone.utc).isoformat()
    manifest = {
        'status':'first_origin_predictions_generated',
        'artifact_id':metadata['artifact_id'],'model_sha256':before,
        'input_sha256':sha(input_file),'generated_at_utc':generated,
        'forecast_origin_utc':'2026-07-01T04:00:00Z',
        'forecast_origin_local':'2026-07-01T00:00:00-04:00',
        'forecast_cases':1578,'zones':263,'horizons':[1,2,3,4,5,6],
        'boosting_rounds':rounds,'inference_device':'cpu',
        'negative_predictions_clipped_to_zero':int((raw<0).sum()),
        'model_retrained':False,'model_file_unchanged':True,
        'actual_pickup_targets_read':False,'accuracy_metrics_calculated':False,
        'full_month_predictions_generated':False,
    }
    (output/'prediction_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    browser = {'artifactId':metadata['artifact_id'],'modelSha256':before,'generatedAtUtc':generated,
               'originUtc':manifest['forecast_origin_utc'],'originIndex':0,'cases':1578,
               'rows':[[int(row.location_id),int(row.forecast_horizon),float(row.prediction)] for row in result.itertuples()]}
    (output/'first-origin-forecasts.json').write_text(json.dumps(browser,separators=(',',':'))+'\n')
    print(json.dumps(manifest,indent=2))
    print('Selected zone 186 hourly forecasts:',result.loc[result.location_id.eq(186),'prediction'].round(2).tolist())


if __name__=='__main__':
    main()
