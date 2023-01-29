import csv
import json
import requests
from datetime import date as dt

# Get today's date to use in downloading current data
today = str(dt.today())


def download_xls(xlsFilePath) -> dict:
    url = f'https://griddata.elia.be/eliabecontrols.prod/interface/fdn/download/availableenergy/volumelevelprice?type=xls&day={today}&csrt=16474748864209014386'

    response = requests.get(url)
    if response.status_code == 200:
        with open(xlsFilePath, "wb") as f:
            f.write(response.content)
    else:
        print('something went wrong!', response.status_code)


def csv_to_json(csvFilePath, jsonFilePath) -> dict:
    jsonArray = []

    # read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf, delimiter=';')

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            jsonArray.append({today: row})

    json_object = json.dumps(jsonArray, indent=4)
    # convert python jsonArray to JSON String and write to file
    with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
        jsonf.write(json_object)

    return json_object


def json_to_csv(jsonObject, csvFilePath):
    headers = [
        'Prediction Datetime',
        'Resolution code',
        'Quarter hour',
        'Input data availability',
        'System imbalance forecast',
        'Probability in [-inf,-400]',
        'Probability in [-400,-200]',
        'Probability in [-200,0]',
        'Probability in [0,200]',
        'Probability in [200,400]',
        'Probability in [400,inf]'
    ]

    with open(csvFilePath, 'w') as csvf:
        writer = csv.writer(csvf)
        count = 0
        for row in jsonObject:
            if count == 0:
                writer.writerow(headers)
                count += 1

            data = {
                "prediction_datetime": row['predictiontimeutc'],
                "resolution_code": row['resolutioncode'],
                "quater_hour": row['predictions_forecastedtimeutc'],
                "input_data_availability": row['predictionquality'],
                "system_imbalance_forecast": row['predictions_silinearregressionforecast'],
                "probability_in_min_inf_min_400": row['predictions_categoricalsiprediction_from_minus_inf_to_minus_400'],
                "probability_in_min_400_min_200": row['predictions_categoricalsiprediction_from_minus_400_to_minus_200'],
                "probability_in_min_200_0": row['predictions_categoricalsiprediction_from_minus_200_to_0'],
                "probability_in_0_200": row['predictions_categoricalsiprediction_from_0_to_200'],
                "probability_in_200_400": row['predictions_categoricalsiprediction_from_200_to_400'],
                "probability_in_400_inf": row['predictions_categoricalsiprediction_from_400_to_inf']
            }
            writer.writerow(data.values())

    print("CSV created!!")


def json_to_csv_078(jsonObject, csvFilePath):
    headers = [
        'Datetime',
        'Quality status',
        'Resolution code',
        'Net regulation volume',
        'System imbalance',
        'Alpha',
        'Marginal incremental price',
        'Marginal decremental price',
        'Positive imbalance price',
        'Negative imbalance price'
    ]

    # Create CSV with the coded headers and discard JSON headers
    with open(csvFilePath, 'w') as csvfile:
        writer = csv.writer(csvfile)
        count = 0
        for row in jsonObject:
            if count == 0:
                writer.writerow(headers)
                count += 1

            writer.writerow(row.values())

    print('CSV created!!')


def csv_to_json_sam(csvFilePath) -> dict:
    jsonArray = []

    # read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            jsonArray.append({today: row})

    return jsonArray
