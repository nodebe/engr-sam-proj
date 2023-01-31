from pulp import *
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import date
from datetime import datetime, timedelta
import requests
import xlrd
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error

from converters import csv_to_json_sam


def get_data_market_clearing(spec_data_market_clearing, i):
    Nbr_product = spec_data_market_clearing.columns.shape[0]
    Nbr_product_one_sided = int(Nbr_product / 2) - 1
    # ,100,100,100,100,100,100]
    Volume_offer_downward = [100 * 0.25 for _ in range(Nbr_product_one_sided)]
    # Last downward offer is very large in volume (avoid infeasibility)
    Volume_offer_downward[-1] = 1000
    # ,100,100,100,100,100,100]
    Volume_offer_upward = [100 * 0.25 for _ in range(Nbr_product_one_sided)]
    # Last upward offer is very large in volume (avoid infeasibility)
    Volume_offer_upward[-1] = 1000
    Price_offer_downward = [spec_data_market_clearing['-100MW'].loc[i], spec_data_market_clearing['-200MW'].loc[i],
                            spec_data_market_clearing['-300MW'].loc[i],
                            spec_data_market_clearing['-400MW'].loc[i], spec_data_market_clearing['-500MW'].loc[i],
                            spec_data_market_clearing['-600MW'].loc[i],
                            spec_data_market_clearing['-700MW'].loc[i], spec_data_market_clearing['-800MW'].loc[i],
                            spec_data_market_clearing['-900MW'].loc[i],
                            spec_data_market_clearing['-1000MW'].loc[i], spec_data_market_clearing['-Max'].loc[i]]
    Price_offer_upward = [spec_data_market_clearing['100MW'].loc[i], spec_data_market_clearing['200MW'].loc[i],
                          spec_data_market_clearing['300MW'].loc[i],
                          spec_data_market_clearing['400MW'].loc[i], spec_data_market_clearing['500MW'].loc[i],
                          spec_data_market_clearing['600MW'].loc[i],
                          spec_data_market_clearing['700MW'].loc[i], spec_data_market_clearing['800MW'].loc[i],
                          spec_data_market_clearing['900MW'].loc[i],
                          spec_data_market_clearing['1000MW'].loc[i], spec_data_market_clearing['Max'].loc[i]]
    return Volume_offer_upward, Volume_offer_downward, Price_offer_upward, Price_offer_downward


def clearing_balancing_energy_market(block_offer_up, block_offer_down, block_cost_up, block_cost_down, Actual_value_NRV):
    # All the optimization program is expressed in MWh

    Actual_value_NRV = float(Actual_value_NRV) * 0.25  # [MWh]
    SI = Actual_value_NRV * 1
    S_r_plus = np.array(block_offer_up)
    S_r_neg = np.array(block_offer_down)
    Lambda_r_plus = np.array(block_cost_up)
    Lambda_r_neg = np.array(block_cost_down)

    N_block_up_reg = S_r_plus.shape[0]
    N_block_down_reg = S_r_neg.shape[0]
    # CREATE THE LP OBJECT, SET UP AS A Minimization PROBLEM --> SINCE WE WANT TO MINIMIZE THE COSTS
    problem_name = 'Imbalance_settlement'
    prob = LpProblem(problem_name, LpMinimize)
    ################################################### DECISION VARIABLES (PROBLEM VARIABLES) ##################################################################
    s_r_plus = pulp.LpVariable.dicts(
        "s_r_plus", (range(N_block_up_reg)), lowBound=0)
    s_r_neg = pulp.LpVariable.dicts(
        "s_r_neg", (range(N_block_down_reg)), lowBound=0)

    ############################################################    CREATE CONSTRAINS ####################################################
    # Constraints linked to clearing imbalance market

    prob += LpConstraint(lpSum([s_r_plus[r_plus] for r_plus in range(N_block_up_reg)])
                         - lpSum([s_r_neg[r_neg]
                                 for r_neg in range(N_block_down_reg)]),
                         rhs=- SI, name='Imb_clearing')
    # Constraints linked to upward_reg
    for r_plus in range(N_block_up_reg):
        prob += - s_r_plus[r_plus] >= -(S_r_plus[r_plus])

    # Constraints linked to Downward_reg
    for r_neg in range(N_block_down_reg):
        prob += -s_r_neg[r_neg] >= -(S_r_neg[r_neg])

    # The objective function is added to 'prob' first
    prob += lpSum([(Lambda_r_plus[r_plus] * s_r_plus[r_plus]) for r_plus in range(N_block_up_reg)]) \
        - lpSum(
        [(Lambda_r_neg[r_neg] * s_r_neg[r_neg]) for r_neg in range(N_block_down_reg)]), "Clearing_balancing_energy_market"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))  # msg=False
    # print("Clearing Market -> Status:", LpStatus[prob.status])
    Price_Imb = prob.constraints['Imb_clearing'].pi
    return Price_Imb


def load_SI_data(Path_excel_file, day, skip, dataset_id):
    be = pd.read_csv(Path_excel_file)

    # be1 = be['Prediction Datetime;Resolution code;Quarter hour;Input data availability;System imbalance forecast;Probability in [-inf']
    # be1 = be1.str.split(';', 20, expand=True)
    be.columns = ['Prediction Datetime', 'Resolution code', 'Quarter hour', 'Input data availability', 'System imbalance forecast', 'Probability in [-inf,-400]',
                  'Probability in [-400,-200]', 'Probability in [-200,0]', 'Probability in [0,200]', 'Probability in [200,400]', 'Probability in [400,inf]']
    be.to_csv(f'new_{dataset_id}.csv', index=False)

    Path = pd.read_csv(f'new_{dataset_id}.csv')
    bee = Path['Prediction Datetime'].str.split('T', 1, expand=True)
    b = bee[0].str.split('-', 3, expand=True)
    bee1 = bee[1].str.split('+', 1, expand=True)
    bee2 = bee1[0].str.split(':', 3, expand=True)
    Path["period1"] = b[2]+'/'+b[1]+'/'+b[0]+' '+bee2[0]+':'+bee2[1]
    Path.drop(['Resolution code', 'Prediction Datetime',
              'Quarter hour'], axis=1, inplace=True)
    Path['period1'] = pd.to_datetime(Path['period1'], format='%d/%m/%Y %H:%M')
    Path.set_index('period1', inplace=True)
    Path = Path.sort_index(axis=0)
    Path = Path.filter(like=day, axis=0)
    PATH = Path[~Path.index.duplicated(keep='first')]
    # PATH

    # PATH = Path

    P1 = PATH.reset_index()
    P1 = P1.drop(['period1'], axis=1)
    format = '%Y-%m-%d'
    mee = datetime.strptime(day, format)
    m1 = []
    for i in range(len(P1)):
        m1.append(mee + timedelta(minutes=i))

    P1['period'] = m1

    P1.set_index('period', inplace=True)

    P1 = P1.interpolate(limit_direction='both')

    daa = []
    for i in range(skip, len(P1), 15):
        x = P1.iloc[i]
        daa.append(x)

    daa = pd.DataFrame(daa)
    daa.columns = ['Input data availability', 'System imbalance forecast', 'Probability in [-inf,-400]',
                   'Probability in [-400,-200]', 'Probability in [-200,0]', 'Probability in [0,200]',
                   'Probability in [200,400]', 'Probability in [400,inf]']
    Pandas_SI_Data = daa.rename_axis('period')

    # print(Pandas_SI_Data)
    return Pandas_SI_Data, len(Pandas_SI_Data)


def load_ARC_data(Path_csv_file, lengt, day, skip):
    # Be careful that the number of columns of the file may change according to the year.
    file_ARC_data = pd.read_excel(Path_csv_file, header=0, skiprows=2)

    # hash the below code to use the one above for curent date time
    date = pd.date_range(start=f'{day} 00:{skip}:00',
                         end=f'{day} 23:59:00', freq='15min')
    date2 = pd.DataFrame(date)
    date2.columns = ['Date']

    file_ARC_data = pd.concat([date2, file_ARC_data], axis=1, join='inner')
    #     file_ARC_data["period"] = file_ARC_data["Date"] + " " + file_ARC_data["Quarter"].str.split(' ').str[0]  # create a timestamp in a correct format
    file_ARC_data["period"] = date2
    file_ARC_data.drop(['Date', 'Quarter'], axis=1, inplace=True)
    file_ARC_data['period'] = pd.to_datetime(file_ARC_data['period'],
                                             format='%Y/%m/%d %H:%M')  # column period in datetime format
    # column Date/Time as index
    file_ARC_data.set_index('period', inplace=True)

    # Fill -MAX column vertically
    file_ARC_data.iloc[:, -1].fillna(method='ffill', axis=0, inplace=True)
    # Fill -MAX column vertically
    file_ARC_data.iloc[:, -1].fillna(method='bfill', axis=0, inplace=True)
    file_ARC_data.iloc[:, 0].fillna(
        method='ffill', axis=0, inplace=True)  # Fill MAX column vertically
    file_ARC_data.iloc[:, 0].fillna(
        method='bfill', axis=0, inplace=True)  # Fill MAX column vertically
    # Fill columns horizontally
    file_ARC_data.interpolate(method='linear', inplace=True, axis=1)

    return file_ARC_data.head(lengt)


def main_program_136(dataset_id):
    print('Started running...')
    global li1, li2, li3, li4, li5, li6, rms, rms1, rms2, rms3, rms4, rms5, final, final1, final2, day_run2, summmary
    li1 = []
    li2 = []
    li3 = []
    li4 = []
    li5 = []
    li6 = []

    today = date.today()

    # for the autodate use the first 'ma'
    ma = str(today)

    qua_end = 0

    Path_csv_SI = f'inputs/volume_{dataset_id}.csv'

    Path_csv_ARC = 'inputs/avail_energy_bal_price.xls'

    # Loading data in dataframes

    Pandas_SI_Data, le = load_SI_data(Path_csv_SI, ma, qua_end, dataset_id)
    Pandas_ARC_Data = load_ARC_data(Path_csv_ARC, le, ma, qua_end)

    for index_data in Pandas_SI_Data.index:
        System_imbalance_forecast = Pandas_SI_Data.loc[index_data,
                                                       'System imbalance forecast']
        block_offer_up, block_offer_down, block_cost_up, block_cost_down = get_data_market_clearing(Pandas_ARC_Data,
                                                                                                    index_data)

        Price_Imb = clearing_balancing_energy_market(
            block_offer_up, block_offer_down,
            block_cost_up, block_cost_down,
            System_imbalance_forecast, )
        Pandas_SI_Data.loc[index_data, 'Constructed_ImbPrice'] = Price_Imb

        li1.append(Price_Imb)
        li2.append(index_data)

    Path = pd.read_csv(f'new_{dataset_id}.csv')
    bee = Path['Prediction Datetime'].str.split('T', 1, expand=True)
    b = bee[0].str.split('-', 3, expand=True)
    bee1 = bee[1].str.split('+', 1, expand=True)
    bee2 = bee1[0].str.split(':', 3, expand=True)
    Path["period1"] = b[2] + '/' + b[1] + '/' + \
        b[0] + ' ' + bee2[0] + ':' + bee2[1]
    #     Path.drop(['Resolution code', 'Quarter hour'], axis=1, inplace=True)
    Path['period1'] = pd.to_datetime(Path['period1'], format='%d/%m/%Y %H:%M')
    Path.set_index('period1', inplace=True)

    Path = Path.sort_index(axis=0)
    Path = Path.filter(like=ma, axis=0)
    PATH = Path[~Path.index.duplicated(keep='first')]

    P1 = PATH.reset_index()
    P1 = P1.drop(['period1'], axis=1)
    format = '%Y-%m-%d'
    mee = datetime.strptime(ma, format)
    m1 = []
    for i in range(len(P1)):
        m1.append(mee + timedelta(minutes=i))

    P1['period'] = m1

    P1.set_index('period', inplace=True)
    daa = []
    # the nth part in the quarter hour should be change here in the case the current values is 13th minute
    for i in range(qua_end, len(P1), 15):
        x = P1.iloc[i]
        daa.append(x)

    daa = pd.DataFrame(daa)
    daa = daa.reset_index()
    daa.drop(['Prediction Datetime', 'index', 'Input data availability', 'Probability in [-inf,-400]',
              'Probability in [-400,-200]', 'Probability in [-200,0]', 'Probability in [0,200]',
              'Probability in [200,400]', 'Probability in [400,inf]'], axis=1, inplace=True)
    daa['System imbalance forecast price'] = li1
    li2 = pd.DataFrame(li2)
    li2.columns = ['Prediction Datetime']

    final = pd.concat([li2, daa], axis=1)
    day_run = str(final['Prediction Datetime'].iloc[-1])
    day_run1 = day_run.replace("-", "_")
    day_run2 = day_run1.replace(":", "_")

    csvFilePath = f'finals/{day_run2}_{str(qua_end)}CQH_imb_price_prediction.csv'
    final.to_csv(csvFilePath, index=False)

    # Create Json
    final_json = csv_to_json_sam(csvFilePath)

    # second simulation
    qua_end = 10

    # Loading data in dataframes
    Pandas_SI_Data, le = load_SI_data(Path_csv_SI, ma, qua_end, dataset_id)
    Pandas_ARC_Data = load_ARC_data(Path_csv_ARC, le, ma, qua_end)

    for index_data in Pandas_SI_Data.index:
        System_imbalance_forecast = Pandas_SI_Data.loc[index_data,
                                                       'System imbalance forecast']
        block_offer_up, block_offer_down, block_cost_up, block_cost_down = get_data_market_clearing(Pandas_ARC_Data,
                                                                                                    index_data)

        Price_Imb = clearing_balancing_energy_market(
            block_offer_up, block_offer_down,
            block_cost_up, block_cost_down,
            System_imbalance_forecast, )
        Pandas_SI_Data.loc[index_data, 'Constructed_ImbPrice'] = Price_Imb

        li3.append(Price_Imb)
        li4.append(index_data)

    Path = pd.read_csv(f'new_{dataset_id}.csv')
    bee = Path['Prediction Datetime'].str.split('T', 1, expand=True)
    b = bee[0].str.split('-', 3, expand=True)
    bee1 = bee[1].str.split('+', 1, expand=True)
    bee2 = bee1[0].str.split(':', 3, expand=True)
    Path["period1"] = b[2] + '/' + b[1] + '/' + \
        b[0] + ' ' + bee2[0] + ':' + bee2[1]

    Path['period1'] = pd.to_datetime(Path['period1'], format='%d/%m/%Y %H:%M')
    Path.set_index('period1', inplace=True)

    Path = Path.sort_index(axis=0)
    Path = Path.filter(like=ma, axis=0)
    PATH = Path[~Path.index.duplicated(keep='first')]

    P1 = PATH.reset_index()
    P1 = P1.drop(['period1'], axis=1)
    format = '%Y-%m-%d'
    mee = datetime.strptime(ma, format)
    m1 = []
    for i in range(len(P1)):
        m1.append(mee + timedelta(minutes=i))

    P1['period'] = m1

    P1.set_index('period', inplace=True)

    daa = []
    # the nth part in the quarter hour should be change here in the case the current values is 13th minute
    for i in range(qua_end, len(P1), 15):
        x = P1.iloc[i]
        daa.append(x)

    daa = pd.DataFrame(daa)
    daa = daa.reset_index()
    daa.drop(['Prediction Datetime', 'index', 'Input data availability', 'Probability in [-inf,-400]',
              'Probability in [-400,-200]', 'Probability in [-200,0]', 'Probability in [0,200]',
              'Probability in [200,400]', 'Probability in [400,inf]'], axis=1, inplace=True)
    daa['System imbalance forecast price'] = li3
    li4 = pd.DataFrame(li4)
    li4.columns = ['Prediction Datetime']
    final1 = pd.concat([li4, daa], axis=1)
    day_run = str(final1['Prediction Datetime'].iloc[-1])
    day_run1 = day_run.replace("-", "_")
    day_run2 = day_run1.replace(":", "_")

    csvFilePath = f'finals/{day_run2}_{str(qua_end)}CQH_imb_price_prediction.csv'
    final1.to_csv(csvFilePath, index=False)

    # Create Json from final1 csv
    final1_json = csv_to_json_sam(csvFilePath)

# third simulation

    qua_end = 5

    # Loading data in dataframes
    Pandas_SI_Data, le = load_SI_data(Path_csv_SI, ma, qua_end, dataset_id)
    Pandas_ARC_Data = load_ARC_data(Path_csv_ARC, le, ma, qua_end)

    for index_data in Pandas_SI_Data.index:
        System_imbalance_forecast = Pandas_SI_Data.loc[index_data,
                                                       'System imbalance forecast']
        block_offer_up, block_offer_down, block_cost_up, block_cost_down = get_data_market_clearing(Pandas_ARC_Data,
                                                                                                    index_data)

        Price_Imb = clearing_balancing_energy_market(
            block_offer_up, block_offer_down,
            block_cost_up, block_cost_down,
            System_imbalance_forecast, )
        Pandas_SI_Data.loc[index_data, 'Constructed_ImbPrice'] = Price_Imb

        li5.append(Price_Imb)
        li6.append(index_data)

    Path = pd.read_csv(f'new_{dataset_id}.csv')
    bee = Path['Prediction Datetime'].str.split('T', 1, expand=True)
    b = bee[0].str.split('-', 3, expand=True)
    bee1 = bee[1].str.split('+', 1, expand=True)
    bee2 = bee1[0].str.split(':', 3, expand=True)
    Path["period1"] = b[2] + '/' + b[1] + '/' + \
        b[0] + ' ' + bee2[0] + ':' + bee2[1]

    Path['period1'] = pd.to_datetime(Path['period1'], format='%d/%m/%Y %H:%M')
    Path.set_index('period1', inplace=True)

    Path = Path.sort_index(axis=0)
    Path = Path.filter(like=ma, axis=0)
    PATH = Path[~Path.index.duplicated(keep='first')]

    P1 = PATH.reset_index()
    P1 = P1.drop(['period1'], axis=1)
    format = '%Y-%m-%d'
    mee = datetime.strptime(ma, format)
    m1 = []
    for i in range(len(P1)):
        m1.append(mee + timedelta(minutes=i))

    P1['period'] = m1

    P1.set_index('period', inplace=True)
    daa = []
    # the nth part in the quarter hour should be change here in the case the current values is 13th minute
    for i in range(qua_end, len(P1), 15):
        x = P1.iloc[i]
        daa.append(x)

    daa = pd.DataFrame(daa)
    daa = daa.reset_index()
    daa.drop(['Prediction Datetime', 'index', 'Input data availability', 'Probability in [-inf,-400]',
              'Probability in [-400,-200]', 'Probability in [-200,0]', 'Probability in [0,200]',
              'Probability in [200,400]', 'Probability in [400,inf]'], axis=1, inplace=True)
    daa['System imbalance forecast price'] = li5
    li6 = pd.DataFrame(li6)
    li6.columns = ['Prediction Datetime']
    final2 = pd.concat([li6, daa], axis=1)
    day_run = str(final2['Prediction Datetime'].iloc[-1])
    day_run1 = day_run.replace("-", "_")
    day_run2 = day_run1.replace(":", "_")

    csvFilePath = f'finals/{day_run2}_{str(qua_end)}CQH_imb_price_prediction.csv'
    final2.to_csv(csvFilePath, index=False)

    # Create Json
    final2_json = csv_to_json_sam(csvFilePath)

    ods078 = pd.read_csv('inputs/volume_ods078.csv')

    summmary = pd.DataFrame(ods078)
    summmary = summmary.drop([len(summmary) - 1])
    bee = summmary['Datetime'].str.split('T', 1, expand=True)
    b = bee[0].str.split('-', 3, expand=True)
    bee1 = bee[1].str.split('+', 1, expand=True)
    bee2 = bee1[0].str.split(':', 3, expand=True)

    summmary["period"] = b[2] + '/' + b[1] + \
        '/' + b[0] + ' ' + bee2[0] + ':' + bee2[1]
    summmary['period'] = pd.to_datetime(
        summmary['period'], format='%d/%m/%Y %H:%M')
    end_date = summmary['period'].iloc[1]
    start_date = summmary['period'].iloc[-1]

    summmary.set_index('period', inplace=True)
    summmary = summmary.sort_index(axis=0)

    main = summmary['System imbalance']
    main1 = summmary['Positive imbalance price']

    cac = final['System imbalance forecast price'].iloc[0:len(summmary)]
    cac1 = final1['System imbalance forecast price'].iloc[0:len(summmary)]
    cac2 = final2['System imbalance forecast price'].iloc[0:len(summmary)]
    main1 = main1.apply(lambda x: float(x))
    cac = cac.apply(lambda x: float(x))
    cac1 = cac1.apply(lambda x: float(x))
    cac2 = cac2.apply(lambda x: float(x))

    calc = final['System imbalance forecast'].iloc[0:len(summmary)]
    calc1 = final1['System imbalance forecast'].iloc[0:len(summmary)]
    calc2 = final2['System imbalance forecast'].iloc[0:len(summmary)]
    main = main.apply(lambda x: float(x))
    calc = calc.apply(lambda x: float(x))
    calc1 = calc1.apply(lambda x: float(x))
    calc2 = calc2.apply(lambda x: float(x))

    te = []
    for i in range(len(summmary)):
        te.append((main.iloc[i] - calc.iloc[i]) / main.iloc[i])

    summmary['error_first'] = te

    te1 = []
    for i in range(len(summmary)):
        te1.append((main.iloc[i] - calc1.iloc[i]) / main.iloc[i])

    summmary['error_second'] = te1

    te2 = []
    for i in range(len(summmary)):
        te2.append((main.iloc[i] - calc2.iloc[i]) / main.iloc[i])

    summmary['error_third'] = te2

    te3 = []
    for i in range(len(summmary)):
        te3.append((main1.iloc[i] - cac.iloc[i]) / main1.iloc[i])

    summmary['error_first_price'] = te3

    te4 = []
    for i in range(len(summmary)):
        te4.append((main1.iloc[i] - cac1.iloc[i]) / main1.iloc[i])

    summmary['error_second_price'] = te4

    te5 = []
    for i in range(len(summmary)):
        te5.append((main1.iloc[i] - cac2.iloc[i]) / main1.iloc[i])

    summmary['error_third_price'] = te5

    rms = (mean_absolute_percentage_error(main, calc))
    rms1 = (mean_absolute_percentage_error(main, calc1))
    rms2 = (mean_absolute_percentage_error(main, calc2))
    rms3 = (mean_absolute_percentage_error(main1, cac))
    rms4 = (mean_absolute_percentage_error(main1, cac1))
    rms5 = (mean_absolute_percentage_error(main1, cac2))

    return {
        'finals': {
            'final': final_json,
            'final_1': final1_json,
            'final_2': final2_json
        },
        'mains': {
            'main': main,
            'main1': main1
        },
        'cacs': {
            'cac': cac,
            'cac1': cac1,
            'cac2': cac2
        },
        'calcs': {
            'calc': calc,
            'calc1': calc1,
            'calc2': calc2
        },
        'rmss': {
            'rms': rms,
            'rms1': rms1,
            'rms2': rms2,
            'rms3': rms3,
            'rms4': rms4,
            'rms5': rms5
        }
    }


if __name__ == '__main__':
    main_program_136('ods136')

# def generate_figs():
#     fig = make_subplots(rows=3,
#                         cols=2,
#                         subplot_titles=(f'1min Volume comparison (MAPE = {rms}%)', f'1min Price comparison (MAPE = {rms3}%)',
#                                         f'5min Volume comparison (MAPE = {rms1}%)', f'5min Price comparison (MAPE = {rms4}%)',
#                                         f'10 min Volume comparison (MAPE = {rms2}%)', f'10min Price comparison (MAPE = {rms5}%)'))

#     fig.add_trace(go.Scatter(x=final['Prediction Datetime'], y=final['System imbalance forecast'],
#                             mode='lines',
#                             name='System imbalance forecast volume 1min',
#                             marker_color='orange'
#                             ), row=1, col=1)
#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['System imbalance'],
#                             mode='lines',
#                             name='Non-validated system imbalance volume',
#                             marker_color='blue'
#                             ), row=1, col=1)

#     fig.add_trace(go.Scatter(x=final['Prediction Datetime'], y=final['System imbalance forecast price'],
#                             mode='lines',
#                             name='System imbalance forecast price 1min',
#                             marker_color='#FF00C0'
#                             ), row=1, col=2)
#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['Positive imbalance price'],
#                             mode='lines',
#                             name='Non-validated system imbalance price',
#                             marker_color='#5CED73'
#                             ), row=1, col=2)

#     fig.add_trace(go.Scatter(x=final1['Prediction Datetime'], y=final1['System imbalance forecast'],
#                             mode='lines',
#                             name='System imbalance forecast volume 5min',
#                             marker_color='yellow'
#                             ), row=2, col=1)

#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['System imbalance'],
#                             mode='lines',
#                             name='Non-validated system imbalance volume',
#                             marker_color='blue'
#                             ), row=2, col=1)
#     fig.add_trace(go.Scatter(x=final['Prediction Datetime'], y=final1['System imbalance forecast price'],
#                             mode='lines',
#                             name='System imbalance forecast price 5min',
#                             marker_color='#A020F0'
#                             ), row=2, col=2)
#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['Positive imbalance price'],
#                             mode='lines',
#                             name='Non-validated system imbalance price',
#                             marker_color='#5CED73'
#                             ), row=2, col=2)

#     fig.add_trace(go.Scatter(x=final2['Prediction Datetime'], y=final2['System imbalance forecast'],
#                             mode='lines',
#                             name='System imbalance forecast volume 10min',
#                             marker_color='red'
#                             ), row=3, col=1)
#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['System imbalance'],
#                             mode='lines',
#                             name='Non-validated system imbalance volume',
#                             marker_color='blue'
#                             ), row=3, col=1)
#     fig.add_trace(go.Scatter(x=final2['Prediction Datetime'], y=final2['System imbalance forecast price'],
#                             mode='lines',
#                             name='system imbalance forecast price 10min',
#                             marker_color='#3A1B2F'
#                             ), row=3, col=2)
#     fig.add_trace(go.Scatter(x=summmary.index, y=summmary['Positive imbalance price'],
#                             mode='lines',
#                             name='Non-validated system imbalance price',
#                             marker_color='#5CED73'
#                             ), row=3, col=2)
#     fig.update_xaxes(title_text="Time")
#     fig.show()


#     # fig.update_xaxes(title_text="Time")
#     # fig.show()
#     fig.write_html(
#         f'finals/{day_run2}_CQHimb_price_prediction.html')


#     fig1 = make_subplots(rows=3,
#                         cols=2,

#                         subplot_titles=('error for 1min volume', 'error for 1min price',
#                                         'error for 5min volume', 'error for 5min price',
#                                         'error for 10min volume', 'error for 10min price',))

#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_first'],
#                             mode='lines',
#                             name=f'error for 1min volume (MAPE = {rms}%)',
#                             marker_color='red'
#                             ), row=1, col=1)
#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_first_price'],
#                             mode='lines',
#                             name=f'error for 1min price (MAPE = {rms3}%)',
#                             marker_color='red'
#                             ), row=1, col=2)

#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_second'],
#                             mode='lines',
#                             name=f'error for 5min volume (MAPE = {rms1}%)',
#                             marker_color='red'
#                             ), row=2, col=1)
#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_second_price'],
#                             mode='lines',
#                             name=f'error for 5min price (MAPE = {rms4}%)',
#                             marker_color='red'
#                             ), row=2, col=2)

#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_third'],
#                             mode='lines',
#                             name=f'error for 5min volume (MAPE = {rms2}%)',
#                             marker_color='red'
#                             ), row=3, col=1)
#     fig1.add_trace(go.Scatter(x=summmary.index, y=summmary['error_third_price'],
#                             mode='lines',
#                             name=f'error for 5min price (MAPE = {rms5}%)',
#                             marker_color='red'
#                             ), row=3, col=2)

#     fig1.update_xaxes(title_text="Time")
#     fig1.show()
#     fig1.write_html(
#         f'finals/{day_run2}_CQHERROR_imb_price_prediction.html')
