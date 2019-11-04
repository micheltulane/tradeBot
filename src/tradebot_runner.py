__author__ = "Michel Tulane"
"""
Main script for "tradebot".
"""

import os
import time
import logging
import json
from tradeBot.src.core.tradebot_worker import Worker
from tradeBot.src.utils.exchanger import PoloExchanger


TRADEBOT_CONFIG_PATH = os.path.abspath("../../local_config/config_tradebot_runner.json")
WORKER_CONFIG_PATH = os.path.abspath("../../local_config/config_worker1.json")
LOGGING_PATH = os.path.abspath("../../logs/tradebot/")
LOGGING_FILENAME = str(int(time.time())) + "_tradebot_runner.log"

# Create and configure logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# Log file handler
fh = logging.FileHandler(LOGGING_PATH + "\\" + LOGGING_FILENAME)
fh.setLevel(logging.INFO)
# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

logger.info("Logger initialized")


# Open config file
with open(TRADEBOT_CONFIG_PATH) as f:
    config = json.load(f)

# Instantiate PoloExchanger
exchange = PoloExchanger(public_key=config["polo_api_public_key"],
                         private_key=config["polo_api_private_key"])

# Instantiate Workers
worker1 = Worker(name="Worker1",
                 config_file_path=WORKER_CONFIG_PATH,
                 logging_path=LOGGING_PATH,
                 exchange=exchange)

# worker1.debug()
worker1.get_balances()
latest_sequence = worker1.poll_graph_til_updated()

pass



