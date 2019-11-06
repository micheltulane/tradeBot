"""
Utility class / python wrapper for Poloniex exchange API
"""

from datetime import datetime
from requests import Request, Session
import hmac
import hashlib
import json
import time
from itertools import count
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
        nonce_counter (counter): Counter object for nonce generation. Initialized with time.time() at _init_
    """
    def __init__(self, public_key, private_key):
        """Initializes a PoloExchanger with a given config file containing the public and private API keys.

        Args:
            public_key (str): Public key for Poloniex API
            private_key (str): Private key for Poloniex API
        """
        self.public_key = public_key
        self.private_key = private_key
        self.nonce_counter = count(int(time.time()*1000))

    @sleep_and_retry
    @limits(calls=POLO_LIMIT_CALLS, period=POLO_LIMIT_PERIOD_S)
    def _call_api(self, prepared_request):
        """Calls the Poloniex API given a request. Rate-limited to POLO_LIMIT_CALLS calls per POLO_LIMIT_PERIOD_S sec.

        Args:
            prepared_request (Request): Prepared Request object
        Returns:
            response (Response): Response object
        """
        s = Session()
        response = s.send(prepared_request)
        if response.status_code != 200:
            raise PoloExchangerError('API response: {}'.format(response.status_code))
        return response

    def _prepare_private_request(self, payload):
        """Prepares a POST request signed with SHA512 for Polo private API

        Args:
            payload (dict): Dict containing POST call parameters
        Returns:
            response (Response): Response object
        """
        headers = {"Key": self.public_key,
                   "Sign": ""}

        payload["nonce"] = next(self.nonce_counter)

        request = Request(
            "POST", POLO_PRIVATE_URL,
            data=payload,
            headers=headers)

        prepared_request = request.prepare()

        signature = hmac.new(bytes(self.private_key, 'utf-8'),
                             bytes(prepared_request.body, 'utf-8'),
                             digestmod=hashlib.sha512)

        prepared_request.headers['Sign'] = signature.hexdigest()

        return self._call_api(prepared_request=prepared_request)

    def _prepare_public_request(self, payload):
        """Prepares a GET request for Polo public API

        Args:
            payload (dict): Dict containing GET call parameters
        Returns:
            response (Response): Response object
        """

        request = Request(
            "GET", POLO_PUBLIC_URL,
            params=payload,)

        prepared_request = request.prepare()
        return self._call_api(prepared_request=prepared_request)

    def return_balances(self):
        """Calls returnBalances from Poloniex API

        Returns:
            Dict containing all balances on the account in format {"currency": "balance"}
        """
        command = "returnBalances"
        payload = {"command": command}
        response = self._prepare_private_request(payload=payload)
        return response.json()

    def return_chart_data(self, currency_pair, period, start, end):
        """Calls returnChartData from Poloniex API

        Args:
            currency_pair (str): The currency pair of the market being requested
            period (str): Candlestick period in seconds. Valid values are 300, 900, 1800, 7200, 14400, and 86400
            start (str): The start of the window in seconds since the unix epoch
            end (str): The end of the window in seconds since the unix epoch

        Returns:
            Dict containing all output fields (refer to Polo API doc)
        """

        command = "returnChartData"
        payload = {"command": command,
                   "currencyPair": currency_pair,
                   "start": start,
                   "end": end,
                   "period": period}

        response = self._prepare_public_request(payload=payload)
        response.encoding = "utf-8"
        return response.json()

    def buy(self, currency_pair, rate, amount, fill_or_kill=True, immediate_or_cancel=True, post_only=False,
            client_order_id=None):
        """BUYS STONKS (Places a limit order in a given market)

        Args:
            currency_pair (str): The currency pair of the market being requested
            rate (float): The rate to purchase one major unit for this trade.
            amount (float): The total amount of minor units offered in this buy order.
            fill_or_kill (bool): Set to True if this order should either fill in its entirety or be completely aborted.
            immediate_or_cancel (bool): True if any part of the order that cannot be filled immed. will be canceled.
            post_only (bool): True if this buy order is only to be placed if no portion of it fills immediately.
            client_order_id (int) 64-bit Integer value used for tracking order across http responses (must be unique)

        Returns:
            Dict containing the order execution info
        """
        command = "buy"
        payload = {"command": command,
                   "currencyPair": currency_pair,
                   "rate": str(rate),
                   "amount": str(amount)}

        # Optional input fields
        if fill_or_kill:
            payload["fillOfKill"] = int(fill_or_kill)
        if immediate_or_cancel:
            payload["immediateOrCancel"] = int(immediate_or_cancel)
        if post_only:
            payload["postOnly"] = int(post_only)
        if client_order_id:
            payload["clientOrderId"] = int(client_order_id)

        response = self._prepare_private_request(payload=payload)
        return response.json()

