__author__ = "Michel Tulane"
"""
Main script for "tradebot".
Instantiates a worker with a given config and runs the worker indefinitely.
"""



from tradeBot.src.core.tradebot_worker import Worker

worker1 = Worker(config_filename="config_worker1.json")

pass



