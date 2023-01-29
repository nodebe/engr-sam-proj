from fastapi import FastAPI
from data_fetcher import get_records, system_imbalance_forecast
from converters import download_xls
from datetime import date as dt
from ods136_vs_078_today_v2 import main_program


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


@app.get('/run_program')
def run_program():
    try:
        # Run this in case the available cron job as not been set
        xlsFilePath = f'inputs/avail_energy_bal_price.xls'
        download_xls(xlsFilePath)

        # Fetch System Imbalance forecast
        system_imbalance_forecast()

        # Get ods078 csv
        get_records()

        # Run the main program.
        final_output = main_program()

        return {'status': 'Success', 'output': final_output}
    except Exception as e:
        return {'status': 'Error', 'message': f'Something went wrong! {e}'}
