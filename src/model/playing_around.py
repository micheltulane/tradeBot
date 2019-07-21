__author__ = "Michel Tulane"

import os
from datetime import datetime
import json
import pandas as pd
from bokeh.plotting import figure, output_file, show, ColumnDataSource

filename = os.path.abspath('C:\\Users\\michel\\PycharmProjects\\tradebot\\tradeBot\\data\\USDT_BTC-300.json')

with open(filename) as json_file:
    data = json.load(json_file)

for i in data:
    dtobj = datetime.fromtimestamp(i["date"])
    i.update({"datetime": dtobj})
    i.update({"day": dtobj.strftime("%d %B, %Y")})
    i.update({"time": dtobj.strftime("%H:%M:%S")})


dataf = pd.DataFrame(data)
source = ColumnDataSource(data=dataf)

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

output_file ("json_exch_data.html")
p = figure(plot_width=1500, plot_height=600, tooltips=TOOLTIPS, tools=tools, toolbar_location="right")

p.line(x='date', y="close", source=source)

show(p)

pass