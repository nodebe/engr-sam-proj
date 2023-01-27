import csv
import json
import requests
from datetime import date as dt

# Get today's date to use in downloading current data
today = str(dt.today())


def download_csv(jsonFilePath, csvFilePath) -> dict:
    url = f'https://griddata.elia.be/eliabecontrols.prod/interface/fdn/download/availableenergy/volumelevelprice?type=csv&day={today}&csrt=16474748864209014386'

    response = requests.get(url)
    if response.status_code == 200:
        with open(csvFilePath, "w", encoding='utf-8') as f:
            f.write(response.text.replace(',', '.'))
    else:
        print('something went wrong!', response.status_code)

    output = csv_to_json(csvFilePath, jsonFilePath)

    return output


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
        'System imbalance forecast',
        'Input data availability',
        'Probability in [-400,-200]',
        'Prediction Datetime',
        'Probability in [400,inf]',
        'Probability in [-inf,-400]',
        'Probability in [-200,0]',
        'Resolution code',
        'Quarter hour',
        'Probability in [0,200]',
        'Probability in [200,400]'
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
