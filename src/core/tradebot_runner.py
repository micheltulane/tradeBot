__author__ = "Michel Tulane"
"""
Main script for "tradebot".
"""


from tradeBot.src.core.tradebot_worker import Worker

worker1 = Worker(config_file_path="../../local_config/config_worker1.json")

worker1.debug()

pass



