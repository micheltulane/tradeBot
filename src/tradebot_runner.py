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
ch.setLevel(logging.INFO)
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
                         private_key=config["polo_api_private_key"],
                         logging_path=LOGGING_PATH)

# Instantiate Workers
worker1 = Worker(name="Worker1",
                 config_file_path=WORKER_CONFIG_PATH,
                 logging_path=LOGGING_PATH,
                 exchange=exchange,
                 worker_budget=100.0)

while True:
    worker1.debug()
# worker1._get_balances()

# book = worker1.exchange.return_order_book(currency_pair="USDT_BTC")
# resultBuy = worker1.exchange.buy(currency_pair="USDT_BTC", rate=9210.0, amount=0.0005, fill_or_kill=True,
#                                  immediate_or_cancel=True, post_only=False, client_order_id=None)
#
# resultSell = worker1.exchange.sell(currency_pair="USDT_BTC", rate=9190.0, amount=0.0005, fill_or_kill=True,
#                                    immediate_or_cancel=True, post_only=False, client_order_id=None)


pass



