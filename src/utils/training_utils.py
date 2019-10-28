__author__ = "Michel Tulane"
# File created 24-JUL-2019


def get_polo_data():
    import pandas as pd
    from datetime import datetime
    import requests
    import json

    start_epoch = 1496188800  # Wednesday 31 May 2017 00:00:00
    one_month = 2592000  # 1 month unix time length
    data_length_months = 30
    data_period = 300  # 5 min
    data_path = "../../data/5min_fetched/"

    # pairs = ["USDT_BTC", "USDT_ETH", "USDT_XRP", "USDT_LTC"]
    pairs = ["USDT_BTC"]
    polo_public_url = "https://poloniex.com/public?"
    cmd_return_chart_data = "returnChartData"

    for pair in pairs:
        start = start_epoch
        data = []
        for month in range(data_length_months):

            end = start + one_month - data_period

            print("Fetch currency pair {} for month {} out of {}".format(pair, month + 1, data_length_months))

            "Fetch currency pair {} for month {} out of {}".format(pair, month + 1, data_length_months)
            reponse = requests.get(
                url=(polo_public_url + "command={cmd}&currencyPair={curr_pair}&start={start}&end={end}&period={period}"
                     .format(cmd=cmd_return_chart_data,
                             curr_pair=pair,
                             start=start,
                             end=end,
                             period=data_period)))
            reponse.encoding = "utf-8"
            temp = json.loads(reponse.content)

            # check if response is the correct length
            if (len(temp) != 8640) and (month < data_length_months-1):
                print("wrong response size. Retrying with increased end epoch")
                reponse = requests.get(
                    url=(
                                polo_public_url + "command={cmd}&currencyPair={curr_pair}&start={start}&end={end}&period={period}"
                                .format(cmd=cmd_return_chart_data,
                                        curr_pair=pair,
                                        start=start,
                                        end=end + data_period,
                                        period=data_period)))
                reponse.encoding = "utf-8"
                temp = json.loads(reponse.content)[:-1]
                print(temp[0])
                print(temp[-1])

            data.extend(temp)
            start = start + one_month

        # generate datetime, day and time values from unix timestamp
        print("Generating metadata and features...")
        for i in data:
            dtobj = datetime.fromtimestamp(i["date"])
            i.update({"day": dtobj.strftime("%d %B, %Y")})
            i.update({"time": dtobj.strftime("%H:%M:%S")})
            i.update({"hour": int(dtobj.strftime("%H"))})
            i.update({"weekday": int(dtobj.strftime("%w"))})
            i.update({(pair + "_price_change_last_5min"): (i["close"] - i["open"])})
            i.update({(pair + "_volatility"): (i["high"] - i["low"])})
            i[(pair + "_volume")] = i.pop("volume")
            i[(pair + "_high")] = i.pop("high")
            i[(pair + "_low")] = i.pop("low")
            i[(pair + "_open")] = i.pop("open")
            i[(pair + "_close")] = i.pop("close")

        print("Generating prediction values")
        for index, i in enumerate(data[:-3]):
            i.update({(pair + "_expected_price_change_15min"): (data[index + 3][pair + "_close"] - i[pair + "_close"])})
            i.update({(pair + "_expected_price_change_5min"): (data[index + 1][pair + "_close"] - i[pair + "_close"])})

        # Convert to dataframe
        data_df = pd.DataFrame(data)

        # Drop last 3 rows
        data_df = data_df.iloc[:-3]

        # save data to excel
        # print("Saving to excel...")
        # data_df.to_excel(data_path + pair + ".xlsx", index=False)

        print("Saving to feather...")
        data_df.to_feather(data_path + pair + ".feather")


get_polo_data()


