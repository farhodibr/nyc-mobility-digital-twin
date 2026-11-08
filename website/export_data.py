"""Export aggregate replay data for a browser; never load or run a model."""
from pathlib import Path
import json
import math
import pandas as pd
import pyarrow.parquet as pq
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent/'dist'
START = pd.Timestamp('2026-06-17')
DATASET = ROOT/'data/gold/simulation/july_2026/dataset'
HISTORY = ROOT/'data/gold/zone_hour_pickups_weather_E_with_concerts.parquet'


def path_from_polygon(poly, bounds):
    xmin,ymin,xmax,ymax=bounds
    scale=min(690/(xmax-xmin),680/(ymax-ymin))
    ox=(740-(xmax-xmin)*scale)/2
    oy=(730-(ymax-ymin)*scale)/2
    def ring(coords):
        pts=[(ox+(x-xmin)*scale,730-oy-(y-ymin)*scale) for x,y in coords]
        return 'M'+'L'.join(f'{x:.1f},{y:.1f}' for x,y in pts)+'Z'
    return ring(poly.exterior.coords)+''.join(ring(interior.coords) for interior in poly.interiors)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=pd.read_parquet(HISTORY,columns=['hour','location_id','total_pickups'],
                        filters=[('hour','>=',START),('hour','<',pd.Timestamp('2026-07-01'))])
    july=pd.read_parquet(ROOT/'data/gold/simulation/july_2026/pilot/zone_hour_pickups_202607.parquet')
    history=pd.concat([old,july],ignore_index=True)
    assert not history.duplicated(['hour','location_id']).any()
    history['hour_index']=((history['hour']-START)/pd.Timedelta(hours=1)).astype(int)
    counts=history.pivot(index='location_id',columns='hour_index',values='total_pickups').reindex(index=range(1,264),columns=range(1080))
    assert not counts.isna().any().any()
    lookup=pd.read_csv(ROOT/'data/external/taxi_zone_lookup.csv').set_index('LocationID')
    dataset=pq.ParquetFile(DATASET/'production_features_202607_simulation.parquet').read(columns=[
        'forecast_origin_utc','forecast_horizon','weather_request_class','location_id',
        'forecast_temperature_f','forecast_precipitation_in','forecast_wind_speed_mph',
        'target_hour_local','is_post_game_demand','is_concert_window','is_wide_concert_window',
        'selected_run_age_hours']).to_pandas()
    dataset['origin_index']=((dataset['forecast_origin_utc']-pd.Timestamp('2026-07-01T04:00Z'))/pd.Timedelta(hours=1)).astype(int)
    grouped=dataset.drop_duplicates(['origin_index','forecast_horizon','weather_request_class']).sort_values(['weather_request_class','origin_index','forecast_horizon'])
    mapping=dataset.drop_duplicates('location_id').set_index('location_id')['weather_request_class']
    weather=[]
    for cls,group in grouped.groupby('weather_request_class',sort=True):
        assert len(group)==738*6
        weather.append({'class':int(cls),'t':[round(float(v),1) for v in group['forecast_temperature_f']],
                        'p':[round(float(v),3) for v in group['forecast_precipitation_in']],
                        'w':[round(float(v),1) for v in group['forecast_wind_speed_mph']]})
    events=dataset.drop_duplicates(['target_hour_local','location_id'])
    events=events[events[['is_post_game_demand','is_concert_window','is_wide_concert_window']].any(axis=1)]
    event_rows=[]
    for row in events.itertuples():
        mask=int(row.is_post_game_demand)+2*int(row.is_concert_window)+4*int(row.is_wide_concert_window)
        event_rows.append([int((row.target_hour_local-START)/pd.Timedelta(hours=1)),int(row.location_id),mask])
    zones=[{'id':i,'name':lookup.loc[i,'Zone'],'borough':lookup.loc[i,'Borough'],'weatherClass':int(mapping.loc[i])} for i in range(1,264)]
    polygons=gpd.read_file(ROOT/'data/external/taxi_zones/extracted/taxi_zones/taxi_zones.shp').to_crs(epsg=2263)
    idcol='LocationID' if 'LocationID' in polygons else 'location_id'
    polygons=polygons[polygons[idcol].between(1,263)].dissolve(by=idcol)
    nyc=polygons.loc[polygons.index!=1]
    bounds=nyc.total_bounds
    paths={}
    for zone,row in nyc.iterrows():
        geometry=row.geometry.simplify(70,preserve_topology=True)
        parts=list(geometry.geoms) if geometry.geom_type=='MultiPolygon' else [geometry]
        paths[str(int(zone))]=''.join(path_from_polygon(poly,bounds) for poly in parts if not poly.is_empty)
    validation=json.loads((DATASET/'dataset_validation.json').read_text())
    document={'startLocal':'2026-06-17T00:00:00','firstOriginLocal':'2026-07-01T00:00:00',
              'originCount':738,'firstOriginHour':336,'hours':1080,'zones':zones,
              'counts':counts.astype(int).values.tolist(),'weather':weather,'events':event_rows,
              'runAges':dataset.drop_duplicates('origin_index').sort_values('origin_index')['selected_run_age_hours'].astype(int).tolist(),
              'mapPaths':paths,'summary':{'rows':validation['rows'],'features':30,'columns':53,'origins':738,'zones':263,'horizons':6},
              'modelStatus':'not_generated','predictionsGenerated':False}
    (OUT/'replay-data.json').write_text(json.dumps(document,separators=(',',':')),encoding='utf-8')
    print(json.dumps({'bytes':(OUT/'replay-data.json').stat().st_size,'zones':len(zones),'mappedZones':len(paths),'predictionsGenerated':False}))


if __name__=='__main__':
    main()
