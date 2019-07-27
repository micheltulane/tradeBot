__author__ = "Michel Tulane"
#File created 24-JUL-2019


# def data_summary(x_train, y_train, x_test, y_test):
#     """Summarize current state of dataset"""
#     print('Train images shape:', x_train.shape)
#     print('Train labels shape:', y_train.shape)
#     print('Test images shape:', x_test.shape)
#     print('Test labels shape:', y_test.shape)
#     print('Train labels:', y_train)
#     print('Test labels:', y_test)

from poloniex import Poloniex
import pandas as pd
from datetime import datetime
import requests
import json

START_EPOCH = 1494547200  # 2017 Mai 12th, 00:00:00 (GMT)
ONE_MONTH = 2592000  # 1 month unix time length
DATA_LENGTH_MONTHS = 27
DATA_PERIOD = 300  # 5 min
DATA_PATH = "../../data/5min_fetched/"

PAIRS = ["USDT_BTC", "USDT_ETH", "USDT_XRP", "USDT_LTC"]
POLO_PUBLIC_URL = "https://poloniex.com/public?"
CMD_RETURN_CHART_DATA = "returnChartData"

for pair in PAIRS:
    start = START_EPOCH
    data = []
    for month in range(DATA_LENGTH_MONTHS):

        end = start + ONE_MONTH - DATA_PERIOD

        print("Fetch currency pair {} for month {} out of {}".format(pair, month+1, DATA_LENGTH_MONTHS))

        "Fetch currency pair {} for month {} out of {}".format(pair, month + 1, DATA_LENGTH_MONTHS)

        reponse = requests.get(
            url=(POLO_PUBLIC_URL + "command={cmd}&currencyPair={curr_pair}&start={start}&end={end}&period={period}"
                 .format(cmd=CMD_RETURN_CHART_DATA,
                         curr_pair=pair,
                         start=start,
                         end=end,
                         period=DATA_PERIOD)))
        reponse.encoding = "utf-8"

        data.extend(json.loads(reponse.content))
        start = start + ONE_MONTH

    # generate datetime, day and time values from unix timestamp
    print("Generating metadata...")
    for i in data:
        dtobj = datetime.fromtimestamp(i["date"])
        i.update({"day": dtobj.strftime("%d %B, %Y")})
        i.update({"time": dtobj.strftime("%H:%M:%S")})

    #  save data to excel
    print("Saving to excel...")
    pd.DataFrame(data).to_excel(DATA_PATH + pair + ".xlsx")
