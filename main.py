from fastapi import FastAPI
from data_fetcher import get_records, system_imbalance_forecast
from converters import download_xls
from datetime import date as dt
from ods136_vs_078_today_v2 import main_program_136
from ods147_vs_078_today import main_program_147


app = FastAPI()


# Will be assigned to cron job to run daily to download grid data
@app.get('/get_grid_data')
def get_grid_data():
    today = str(dt.today())
    xlsFilePath = f'inputs/avail_energy_bal_price.xls'
    try:
        run_download = download_xls(xlsFilePath)
        return {'status': 'Success!'}

    except Exception as e:
        return {'status': 'Error', 'message': f'Something went wrong! {e}'}


@app.get('/run_program_136')
def run_program_136():
    try:
        dataset_id = 'ods136'
        # Run this in case the available cron job as not been set
        xlsFilePath = f'inputs/avail_energy_bal_price.xls'
        download_xls(xlsFilePath)

        # Fetch System Imbalance forecast
        system_imbalance_forecast(dataset_id)

        # Get ods078 csv
        get_records()

        # Run the main program.
        final_output = main_program_136(dataset_id)

        return {'status': 'Success', 'output': final_output}
    except Exception as e:
        return {'status': 'Error', 'message': f'Something went wrong! {e}'}


@app.get('/run_program_147')
def run_program_147():
    try:
        dataset_id = 'ods147'
        # Run this in case the available cron job as not been set
        xlsFilePath = f'inputs/avail_energy_bal_price.xls'
        download_xls(xlsFilePath)

        # Fetch System Imbalance forecast
        system_imbalance_forecast(dataset_id)

        # Get ods078 csv
        get_records()

        # Run the main program.
        final_output = main_program_147(dataset_id)

        return {'status': 'Success', 'output': final_output}
    except Exception as e:
        raise e
        return {'status': 'Error', 'message': f'Something went wrong! {e}'}
