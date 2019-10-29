__author__ = "Michel Tulane"
"""
Utility class / python wrapper for crypto exchanges APIs
"""

from datetime import datetime
from requests import Request, Session
import hmac
import hashlib
import json
from ratelimit import limits, sleep_and_retry

POLO_PRIVATE_URL = "https://poloniex.com/tradingApi"
POLO_PUBLIC_URL = "https://poloniex.com/public"
POLO_LIMIT_CALLS = 6
POLO_LIMIT_PERIOD_S = 1

class PoloExchangerError(Exception):
    """Exception raised from the PoloExchanger class
    Attributes:
        message -- explanation of the error
    """
    def __init__(self, mesg):
        self.mesg = mesg

    def __repr__(self):
        return self.mesg

    def __str__(self):
        return self.mesg


class PoloExchanger:
    """PoloExchanger interfaces with the Poloniex HTTP API using the request library.

    Attributes:
        public_key (str): Public key for Poloniex API
        private_key (str): Private key for Poloniex API
    """
    def __init__(self, public_key, private_key):
        """Initializes a PoloExchanger with a given config file containing the public and private API keys.

        Args:
            public_key (str): Public key for Poloniex API
            private_key (str): Private key for Poloniex API
        """
        self.public_key = public_key
        self.private_key = private_key

    @sleep_and_retry
    @limits(calls=POLO_LIMIT_CALLS, period=POLO_LIMIT_PERIOD_S)
    def _call_api(self, prepared_request):
        """Calls the Poloniex API given a request. Rate-limited to POLO_LIMIT_CALLS calls per POLO_LIMIT_PERIOD_S sec.

        Args:
            request (Request): Prepared Request object

        Returns:
            response(Response): Response object
        """
        s = Session()
        response = s.send(prepared_request)
        if response.status_code != 200:
            raise PoloExchangerError('API response: {}'.format(response.status_code))
        return response
