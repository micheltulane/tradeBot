__author__ = "Michel Tulane"
"""
Main script for "tradebot".
"""

TRADEBOT_CONFIG_PATH = "../../local_config/config_tradebot_runner.json"
WORKER_CONFIG_PATH = "../../local_config/config_worker1.json"

import os
import json
from tradeBot.src.core.tradebot_worker import Worker
from tradeBot.src.utils.exchanger import PoloExchanger

# Open config file
TRADEBOT_CONFIG_PATH = os.path.abspath(TRADEBOT_CONFIG_PATH)
with open(TRADEBOT_CONFIG_PATH) as f:
    config = json.load(f)

# Create PoloExchanger instance
exchange = PoloExchanger(public_key=config["polo_api_public_key"],
                         private_key=config["polo_api_private_key"])

worker1 = Worker(config_file_path=WORKER_CONFIG_PATH, exchange=exchange)

worker1.debug()

pass



