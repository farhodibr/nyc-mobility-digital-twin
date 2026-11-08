"""Acquire and validate July 2026 HVFHV data without replacing existing files."""
from pathlib import Path
import json
import urllib.request
import pyarrow.parquet as pq
import duckdb

URL = 'https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2026-07.parquet'
TARGET = Path(__file__).resolve().parents[1] / 'data/bronze/fhvhv_tripdata_2026-07.parquet'
REPORT = Path(__file__).resolve().parent / 'july_tlc_validation.json'


def main():
    if not TARGET.exists():
        partial = TARGET.with_suffix('.parquet.part')
        if partial.exists():
            print('Checking the previously downloaded Parquet before finalizing it.', flush=True)
        else:
            request = urllib.request.Request(URL, headers={'User-Agent': 'NYC-Capstone-Data-Download/1.0'})
            size = 0
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('xb') as output:
                expected = response.headers.get('Content-Length')
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
                    size += len(chunk)
                    if size % (32 * 1024 * 1024) == 0:
                        print(f'Downloaded {size / 1024**2:.0f} MiB', flush=True)
            if expected and size != int(expected):
                raise ValueError(f'Incomplete download: {size} of {expected} bytes')
        with pq.ParquetFile(partial) as parquet:
            required = {'pickup_datetime', 'PULocationID'}
            if not required.issubset(parquet.schema_arrow.names):
                raise ValueError('Required pickup columns are missing; partial file retained.')
        partial.rename(TARGET)
        print(f'Saved {TARGET}', flush=True)
    else:
        print('Existing July file retained; validating without replacing it.', flush=True)

    parquet = pq.ParquetFile(TARGET)
    con = duckdb.connect(':memory:')
    con.execute("SET memory_limit='1GB'")
    summary = con.execute('''
        SELECT count(*) AS total_rows,
               min(pickup_datetime) AS first_pickup,
               max(pickup_datetime) AS last_pickup,
               count(*) FILTER (WHERE pickup_datetime IS NULL) AS missing_pickup_time,
               count(*) FILTER (WHERE PULocationID IS NULL) AS missing_pickup_zone,
               count(*) FILTER (WHERE pickup_datetime < TIMESTAMP '2026-07-01'
                                  OR pickup_datetime >= TIMESTAMP '2026-08-01') AS outside_july
        FROM read_parquet(?)
    ''', [str(TARGET)]).fetchone()
    daily = con.execute('''
        SELECT CAST(pickup_datetime AS DATE) AS pickup_date, count(*) AS trips
        FROM read_parquet(?)
        WHERE pickup_datetime >= TIMESTAMP '2026-07-01'
          AND pickup_datetime < TIMESTAMP '2026-08-01'
        GROUP BY 1 ORDER BY 1
    ''', [str(TARGET)]).fetchall()
    con.close()
    report = {
        'source_url': URL,
        'local_file': str(TARGET),
        'file_bytes': TARGET.stat().st_size,
        'parquet_rows': parquet.metadata.num_rows,
        'column_count': len(parquet.schema_arrow.names),
        'total_rows': summary[0],
        'first_pickup': str(summary[1]),
        'last_pickup': str(summary[2]),
        'missing_pickup_time': summary[3],
        'missing_pickup_zone': summary[4],
        'outside_july': summary[5],
        'july_dates_present': len(daily),
        'daily_trip_counts': [{'date': str(day), 'trips': count} for day, count in daily],
        'model_retrained': False,
    }
    REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'daily_trip_counts'}, indent=2), flush=True)
    if len(daily) != 31:
        raise ValueError('Not all 31 July dates are represented; inspect the report.')
    if summary[0] != parquet.metadata.num_rows:
        raise ValueError('Parquet metadata and scanned row count disagree.')


if __name__ == '__main__':
    main()
