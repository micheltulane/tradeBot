__author__ = "Michel Tulane"
"""
Class description file for a tradebot Worker
"""

import os
import numpy
import json
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
        model_json_filename (str): Name of keras model config json file.
        saved_weights_filename (str): Name of keras model weights file.
        currency_pair (str): The currency pair on which the model was trained
        buy_threshold (float): Model prediction output threshold to buy currency
        sell_threshold (float): Model prediction output threshold to sell currency
        model (Sequential): Sequential neural network model
        exchanger (PoloExchanger): Poloniex API wrapper object
    """

    def __init__(self, config_file_path):
        """Initializes a Worker with a given config file, also loads pre-trained model and weights.

        Args:
            config_file_path (str): Path to the Worker's config file.
        """
        # Load config file...
        config_file_path = os.path.abspath(config_file_path)
        config_path = os.path.dirname(config_file_path)

        with open(config_file_path) as f:
            config = json.load(f)

        self.model_json_filename = os.path.abspath(config_path + "\\" + config["model_json_filename"])
        self.saved_weights_filename = os.path.abspath(config_path + "\\" + config["model_weights_filename"])
        self.currency_pair = config["currency_pair"]
        self.buy_threshold = config["buy_threshold"]
        self.sell_threshold = config["sell_threshold"]

        # Load model from JSON
        with open(self.model_json_filename, 'r') as json_file:
            self.model = json_file.read()
            json_file.close()
        self.model = model_from_json(self.model)

        # Load weights from file
        self.model.load_weights(self.saved_weights_filename)

        # Create PoloExchanger instance
        self.exchanger = PoloExchanger(public_key=config["polo_api_public_key"],
                                       private_key=config["polo_api_private_key"])

    def debug(self):
        # test = self.exchanger.return_balances()
        test = self.exchanger.return_chart_data(currency_pair="USDT_BTC",
                                                period="300",
                                                start="1571149212",
                                                end="1572827612")
        print(test[-1]["date"])

        pass


