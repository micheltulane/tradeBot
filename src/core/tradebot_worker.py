__author__ = "Michel Tulane"
"""
Class description file for a tradebot Worker
"""

import os
import logging
import time
from datetime import datetime
import threading
import numpy
import json
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.layers import Dense, CuDNNLSTM
from tradeBot.src.utils.exchanger import PoloExchanger


class WorkerError(Exception):
    """Exception raised from the Worker class
    Attributes:
        message -- explanation of the error
    """
    def __init__(self, mesg):
        self.mesg = mesg

    def __repr__(self):
        return self.mesg

    def __str__(self):
        return self.mesg


class Worker:
    """A Worker uses a pre-trained lstm neural network to predict the price of a currency and buy/sell accordingly.

    Attributes:
        name (str): Name of the Worker
        model_json_filename (str): Name of keras model config json file.
        saved_weights_filename (str): Name of keras model weights file.
        currency_pair (str): The currency pair on which the model was trained
        buy_threshold (float): Model prediction output threshold to buy currency
        sell_threshold (float): Model prediction output threshold to sell currency
        model (Sequential): Sequential neural network model
        exchange (PoloExchanger): Poloniex API wrapper object
        logger (Logger): Module-level logger instance
        minor_currency (str): Name of minor currency (ex: USDT)
        minor_balance (Float): Amount of the minor currency in account
        major_currency (str): Name of minor currency (ex: BTC)
        major_balance (Float): Amount of the major currency in account
        data_period (str): Candlestick data period, in seconds (ex: 300)
        model_sequence_length (int): Number of points used for inference
        logging_path (str): Path to the Worker's log files
    """

    def __init__(self, name, config_file_path, logging_path, exchange, worker_budget=None):
        """Initializes a Worker with a given config file, also loads pre-trained model and weights.

        Args:
            config_file_path (str): Path to the Worker's config file.
            logging_path (str): Path to the Worker's log files
            exchange (PoloExchanger): Poloniex API wrapper object
            worker_budget (float): Max amount traded at any time, in minor currency
        """
        self.name = name
        self.logger = logging.getLogger(__name__)
        self.logging_path = logging_path
        self.worker_budget = worker_budget
        # Load config file...
        config_file_path = os.path.abspath(config_file_path)
        config_dir = os.path.dirname(config_file_path)

        self.logger.info(
            "Initializing Worker with config file: {config}".format(config=config_file_path))

        with open(config_file_path) as f:
            config = json.load(f)

        self.model_json_filename = os.path.abspath(config_dir + "\\" + config["model_json_filename"])
        self.saved_weights_filename = os.path.abspath(config_dir + "\\" + config["model_weights_filename"])
        self.currency_pair = config["currency_pair"]
        self.buy_threshold = config["buy_threshold"]
        self.sell_threshold = config["sell_threshold"]
        self.data_period = config["data_period"]
        self.model_sequence_length = config["model_sequence_length"]

        # Load model from JSON
        with open(self.model_json_filename, 'r') as json_file:
            self.model = json_file.read()
            json_file.close()
        self.model = model_from_json(self.model)

        # Load weights from file
        self.model.load_weights(self.saved_weights_filename)

        # Save PoloExchanger instance
        self.exchange = exchange

        # Init misc attributes
        self.minor_balance = None
        self.major_balance = None
        self.minor_currency, self.major_currency = self.currency_pair.split("_")

    def get_balances(self):
        """Gets current balances in account and for major and minor currencies in the pair.
        """
        self.logger.info("Getting Balances...")

        balances = self.exchange.return_balances()

        self.minor_balance = float(balances[self.minor_currency])
        self.major_balance = float(balances[self.major_currency])

        self.logger.info("Minor ({min_curr}): {min_bal}, Major ({maj_curr}): {maj_bal}".format(
            min_curr=self.minor_currency,
            min_bal=self.minor_balance,
            maj_curr=self.major_currency,
            maj_bal=self.major_balance))

    def poll_graph_til_updated(self):
        """Gets the last [sequence length] numbers of candlestick data from Poloniex. Loop until a new candlestick is
        available and return the most up to date sequence.
        """
        self.logger.info("Polling graph until updated...")
        # Get current latest candlestick (only get latest 10 points to minimize payloads)
        now = int(time.time())
        temp = self.exchange.return_chart_data(
            currency_pair=self.currency_pair,
            period=self.data_period,
            start=str(now-int(self.data_period)*10),
            end=str(now))
        last_candle_timestamp = int(temp[-1]["date"])

        # Wait until the latest candlestick point is updated, poll every second
        while True:
            now = int(time.time())
            temp = self.exchange.return_chart_data(
                currency_pair=self.currency_pair,
                period=self.data_period,
                start=str(now-int(self.data_period)*10),
                end=str(now))

            time.sleep(1)
            self.logger.debug(
                "last_candle_timestamp: {}, new_timestamp: {}".format(last_candle_timestamp, int(temp[-1]["date"])))
            if int(temp[-1]["date"]) > last_candle_timestamp:
                # We have a new data point to process
                break

        # Return full sequence
        return self.exchange.return_chart_data(
            currency_pair=self.currency_pair,
            period=self.data_period,
            start=str(now-int(self.data_period)*(self.model_sequence_length-1)),
            end=str(int(time.time())))

    def prepare_data(self, chart_data):
        """Takes a given chart data point list (list of dicts), generates features and metadata for model inference

        Args:
            chart_data (list): List of dicts type containing candlestick chart data

        Returns:
            (Dataframe) Contains processed chart data plus new generated features
        """
        self.logger.info("Preparing data...")
        for i in chart_data:
            dtobj = datetime.fromtimestamp(i["date"])
            i.update({"day": dtobj.strftime("%d %B, %Y")})
            i.update({"time": dtobj.strftime("%H:%M:%S")})
            i.update({"hour": int(dtobj.strftime("%H"))})
            i.update({"weekday": int(dtobj.strftime("%w"))})
            i.update({(self.currency_pair + "_price_change_last_5min"): (i["close"] - i["open"])})
            i.update({(self.currency_pair + "_volatility"): (i["high"] - i["low"])})
            i[(self.currency_pair + "_volume")] = i.pop("volume")
            i[(self.currency_pair + "_high")] = i.pop("high")
            i[(self.currency_pair + "_low")] = i.pop("low")
            i[(self.currency_pair + "_open")] = i.pop("open")
            i[(self.currency_pair + "_close")] = i.pop("close")

        return pd.DataFrame(chart_data)

    def do_prediction(self, prepared_data):
        """Takes prepared chart data (with additional features and metadata) and runs inference with it.

        Args:
            prepared_data (Dataframe): Processed chart data plus new generated features

        Returns:
            (array) Predicted values (entire sequence length)
        """
        self.logger.info("Making prediction...")

        # Dropping unnecessary Series from input dataframe (human-readable stuff)
        x_data = prepared_data.drop(columns=["day", "time"])

        # Converting to numpy array
        x_data_ndarray = x_data.values

    def do_after_prediction(self):
        pass

    def debug(self):
        # test = self.exchanger.return_balances()
        test = self.exchange.return_chart_data(currency_pair="USDT_BTC",
                                               period="300",
                                               start="1571149212",
                                               end="1572827612")
        print(test[-1]["date"])

        pass


