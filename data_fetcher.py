import requests
from datetime import datetime as dt
from datetime import date
from converters import json_to_csv, json_to_csv_078


def get_records(dataset_id='ods078'):
    url = f'https://opendata.elia.be/api/v2/catalog/datasets/{dataset_id}/records?order_by=datetime%20desc&limit=20&offset=0&lang=en&timezone=UTC'
    csvFilePath = f'open_data_csvs/volume_ods078.csv'

    fetch_data = requests.get(url)

    records = fetch_data.json()['records']
    record_fields = []

    for record in records:
        record_fields.append(record['record']['fields'])

    # Convert json to csv and save
    json_to_csv_078(record_fields, csvFilePath)

    return record_fields


def system_imbalance_forecast(dataset_id='ods136'):
    begin_time = date.today()
    end_time = dt.now().strftime('%Y-%m-%dT%H:%M')
    csvFilePath = f'system_imbalance_csvs/new_ods136.csv'

    url = f'https://opendata.elia.be/api/records/1.0/search/?dataset={dataset_id}&q=predictiontimeutc%3A[{begin_time}+TO+{end_time}]&rows=1500&sort=predictiontimeutc&facet=predictiontimeutc&facet=resolutioncode&facet=predictions_forecastedtimeutc'

    fetch_data = requests.get(url)
    records = fetch_data.json()['records']
    record_fields = []

    for record in records:
        record_fields.append(record['fields'])

    # Convert json to csv and save
    json_to_csv(record_fields, csvFilePath)

    return record_fields
