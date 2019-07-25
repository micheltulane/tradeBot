__author__ = "Michel Tulane"

import os
from datetime import datetime
import random
import json
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_file, show
from bokeh.layouts import column


# DATA_PATH = "../../data/15min_json/"
DATA_PATH = "../../data/5min_json/"

COLOR_PALETTE = \
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

PLOT_WIDTH = 1500
PLOT_HEIGHT = 350


# CURRENCY_PAIRS = [
#     {"name": "USDT_BTC",
#      "base_currency": "USDT",
#      "counter_currency": "BTC",
#      "json_file_name": "USDT_BTC-900.json",
#      },
#     {"name": "USDT_ETH",
#      "base_currency": "USDT",
#      "counter_currency": "ETH",
#      "json_file_name": "USDT_ETH-900.json",
#      },
#     {"name": "USDT_LTC",
#      "base_currency": "USDT",
#      "counter_currency": "LTC",
#      "json_file_name": "USDT_LTC-900.json",
#      },
#     {"name": "USDT_XRP",
#      "base_currency": "USDT",
#      "counter_currency": "XRP",
#      "json_file_name": "USDT_XRP-900.json",
#      }
# ]

CURRENCY_PAIRS = [
    {"name": "USDT_BTC",
     "base_currency": "USDT",
     "counter_currency": "BTC",
     "json_file_name": "USDT_BTC-300.json",
     },
    {"name": "USDT_ETH",
     "base_currency": "USDT",
     "counter_currency": "ETH",
     "json_file_name": "USDT_ETH-300.json",
     },
    {"name": "USDT_LTC",
     "base_currency": "USDT",
     "counter_currency": "LTC",
     "json_file_name": "USDT_LTC-300.json",
     }
]

# Parse currency pairs and load / generate data
for currency_pair in CURRENCY_PAIRS:

    # Load json data
    with open(os.path.abspath(DATA_PATH + currency_pair["json_file_name"])) as json_file:
        data = json.load(json_file)

        # generate datetime, day and time values from unix timestamp
        for i in data:
            dtobj = datetime.fromtimestamp(i["date"])
            i.update({"datetime": dtobj})
            i.update({"day": dtobj.strftime("%d %B, %Y")})
            i.update({"time": dtobj.strftime("%H:%M:%S")})

        # Append data to currency pair dict
        currency_pair.update({"data": data})


# Plot data using Bokeh

TOOLTIPS = [
    ("index", "$index"),
    ("unix date", "@date"),
    ("day", "@day"),
    ("time", "@time"),
    ("high", "@high"),
    ("low", "@low"),
    ("open", "@open"),
    ("close", "@close"),
    ("volume", "@volume"),
    ("quoteVolume", "@quoteVolume"),
    ("weightedAverage", "@weightedAverage")
]

tools = "pan,wheel_zoom,box_zoom,reset"

output_file("../../../json_exch_data.html")

# Generate line graphs for all currency pairs and append to currency pair dict
for i, currency_pair in enumerate(CURRENCY_PAIRS):

    # Convert dict data to dataframe
    dataframe = pd.DataFrame(currency_pair["data"])
    source = ColumnDataSource(data=dataframe)

    # Save to excel
    # dataframe.to_excel(DATA_PATH + currency_pair["json_file_name"] + ".xlsx")

    # Create figure
    # It it's the first figure, do not link x_range with previous
    if i == 0:
        p = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, title=currency_pair["name"],
                   tooltips=TOOLTIPS, tools=tools, toolbar_location="right")
        first_figure = p
    else:
        p = figure(plot_width=PLOT_WIDTH, plot_height=PLOT_HEIGHT, title=currency_pair["name"],
                   tooltips=TOOLTIPS, tools=tools, toolbar_location="right",
                   x_range=first_figure.x_range)

    # Create line graph
    p.line(x='date', y="close", color=COLOR_PALETTE[i], source=source)

    # Append to currency pair dict
    currency_pair.update({"figure": p})


# Show all figures in column
page = column([currency_pair['figure'] for currency_pair in CURRENCY_PAIRS], sizing_mode="scale_both")

show(page)

pass