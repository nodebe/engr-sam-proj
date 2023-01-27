from fastapi import FastAPI
from data_fetcher import get_records, system_imbalance_forecast
from converters import download_csv
from datetime import date as dt


app = FastAPI()


@app.get('/get_open_data')
def get_open_data():
    data = get_records()

    return {'data': data}


@app.get('/get_system_imbalance')
def get_system_imbalance():

    data = system_imbalance_forecast()

    return {'data': data}

# Will be assigned to cron job


@app.get('/get_grid_data')
def get_grid_data():
    today = str(dt.today())
    jsonFilePath = f'available_energy_jsons/{today}.json'
    csvFilePath = f"available_energy_csvs/{today}.csv"
    try:
        run_download = download_csv(jsonFilePath, csvFilePath)
        return {'status': 'Success!'}

    except Exception as e:
        return {'status': 'Error', 'message': f'Something went wrong! {e}'}


@app.get('/run_program')
def run_program():
    # Update the system imbalance csv sheet
    system_imbalance_data = system_imbalance_forecast()

    # Update the 078 csv sheet
    open_data_records = get_records()
    print('Done fetching, time to calculate...')
    # Calls the model for calculation
    import ods136_vs_078_today
