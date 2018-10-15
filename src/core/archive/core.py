# File created 27-DEC-2017
# Author: Michel Tulane

# using Python API wrapper for Poloniex: https://github.com/s4w3d0ff/python-poloniex/archive/v0.4.7.zip

from src.utils import poloniexutils
from poloniex import Poloniex, Coach


#Init CSV logging
import csv
from time import gmtime, strftime

filename = ('tradeBot_log_' + strftime("%a_%d_%b_%Y_%H%M%S", gmtime()) + '.csv')
with open(filename, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['fwdMargin'] + ['revMargin'])

    # init Poloniex wrapper
    myCoach = Coach(timeFrame=1.0, callLimit=6) # create a coach, limit = 6 API calls/s
    polo = Poloniex(key='1L72WO53-T96VC9QW-EUWB0KQN-SNU0LRUC',secret='51ce4fb70439a7c2b1457bcedcd4528ec4598d7d317577288b7eb65f50df7b6b571db6d3bdc14a48eea885bf7aad5be50b1bf843345aebf99a0dede04c8de746', coach=myCoach)

    # Get current fees and volume
    print("Getting current fees...")
    fees = polo('returnFeeInfo')
    lastMakerFee = float(fees['makerFee'])
    lastTakerFee = float(fees['takerFee'])
    last30dayVolume = float(fees['thirtyDayVolume'])

    # Print current fees and volume
    print("Maker fee: %f %% " % (lastMakerFee*100))
    print("Taker fee: %f %% " % (lastTakerFee*100))
    print("30 day volume %f BTC" % last30dayVolume)

    # Get balances
    print("\nGetting current balances...")
    balance = polo('returnBalances')

    # Print balances if not 0
    for key, val in balance.items():
        if float(val) != 0:
            print("%s " % val, key)

    # market tags
    usdtMarkets = ['USDT_BCH', 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH', 'USDT_LTC', 'USDT_NXT', 'USDT_REP', 'USDT_STR', 'USDT_XMR', 'USDT_XRP', 'USDT_ZEC']
    btcMarkets = ['BTC_BCH', 'BTC_BCN', 'BTC_DASH', 'BTC_ETC', 'BTC_ETH', 'BTC_LTC', 'BTC_NXT', 'BTC_REP', 'BTC_STR', 'BTC_XMR', 'BTC_XRP', 'BTC_ZEC']
    ethMarkets = ['ETH_BCH', 'ETH_CVC', 'ETH_ETC', 'ETH_GAS', 'ETH_GNO', 'ETH_GNT', 'ETH_LSK', 'ETH_OMG', 'ETH_REP', 'ETH_STEEM', 'ETH_ZEC', 'ETH_ZRX']
    xmrMarkets = ['XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC', 'XMR_MAID', 'XMR_NXT', 'XMR_ZEC']

    # triangle loops definitions
    triangleLoops = [   [['USDT_BCH', 'buy'], ['BTC_BCH', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_BTC', 'buy'], ['BTC_BCH', 'buy'], ['USDT_BCH', 'sell']],    \
                        [['USDT_DASH', 'buy'], ['BTC_DASH', 'sell'], ['USDT_BTC', 'sell']], \
                        [['USDT_ETC', 'buy'], ['BTC_ETC', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_ETH', 'buy'], ['BTC_ETH', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_LTC', 'buy'], ['BTC_LTC', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_NXT', 'buy'], ['BTC_NXT', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_REP', 'buy'], ['BTC_REP', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_STR', 'buy'], ['BTC_STR', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_XMR', 'buy'], ['BTC_XMR', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_XRP', 'buy'], ['BTC_XRP', 'sell'], ['USDT_BTC', 'sell']],   \
                        [['USDT_ZEC', 'buy'], ['BTC_ZEC', 'sell'], ['USDT_BTC', 'sell']],
                        #[['USDT_XMR', 'buy'], ['XMR_BCN', 'buy'], ['BTC_BCN', 'sell']]   ] marche pas
                        ]

    botError = 0
    while botError==0:
        
        # loop through defined trading triangles
        print("\nLooping through defined trading triangles...")
        for i, tradeLoop in enumerate(triangleLoops):
            bookInfos = [None, None, None]
            temp = 0
            fwdMargin = 0
            revMargin = 0

            print('\nTrade triangle no.' + repr(i))
            # get prices according to trade definition (sell or buy)
            for j, operation in enumerate(tradeLoop):
                bookInfos[j] = poloniexutils.getBookInfo(polo.returnOrderBook(operation[0]))
                print('Market no.%s: %s' % (repr(j), operation[0]))

            # Calculate forward loop margin...................................
            # operation 1
            if tradeLoop[0][1] == 'buy':
                temp = 1/(bookInfos[0]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[0][1] == 'sell':
                temp = 1*(midPrice*(1-lastTakerFee))
    
            # operation 2
            if tradeLoop[1][1] == 'buy':
                temp = temp/(bookInfos[1]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[1][1] == 'sell':
                temp = temp*(bookInfos[1]['bidPrice']*(1-lastTakerFee))

            # operation 3
            if tradeLoop[2][1] == 'buy':
                temp = temp/(bookInfos[2]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[2][1] == 'sell':
                temp = temp*(bookInfos[2]['bidPrice']*(1-lastTakerFee))

            fwdMargin = temp-1
            print('Forward loop profit margin: %f %%' % (fwdMargin*100))
            
            # Calculate reverse loop margin..................................
            # operation 1
            if tradeLoop[2][1] == 'sell':
                temp = 1/(bookInfos[2]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[2][1] == 'buy':
                temp = 1*(bookInfos[2]['bidPrice']*(1-lastTakerFee))
    
            # operation 2
            if tradeLoop[1][1] == 'sell':
                temp = temp/(bookInfos[1]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[1][1] == 'buy':
                temp = temp*(bookInfos[1]['bidPrice']*(1-lastTakerFee))

            # operation 3
            if tradeLoop[0][1] == 'sell':
                temp = temp/(bookInfos[0]['askPrice']*(1+lastTakerFee))
            elif tradeLoop[0][1] == 'buy':
                temp = temp*(bookInfos[0]['bidPrice']*(1-lastTakerFee))

            revMargin = temp-1

            # log results
     
            spamwriter.writerow([str(fwdMargin)] + [str(revMargin)] + [strftime("%H:%M:%S", gmtime())])
            print('Reverse loop profit margin: %f %%' % (revMargin*100))
            
        # end triangles loop

        # loop through USDT markets
        '''
        print("\nLooping through USDT markets...")
        for market in usdtMarkets:
            book = polo.returnOrderBook(market)
            infoBook = utils.getBookInfo(book)
            #temp = (infoBook['askPrice']-infoBook['bidPrice'])/infoBook['askPrice']
            temp = (infoBook['askPrice']/(1+lastTakerFee)-infoBook['bidPrice']/(1-lastTakerFee))/(infoBook['askPrice']/(1+lastTakerFee))

            print("Market: %s,\tBID: %f,\tASK: %f,\tMargin: %f %%" % (market, infoBook['bidPrice'], infoBook['askPrice'], temp*100))

        #loop through BTC markets
        print("\nLooping through BTC markets...")
        for market in btcMarkets:
            book = polo.returnOrderBook(market)
            infoBook = utils.getBookInfo(book)
            #temp = (infoBook['askPrice']-infoBook['bidPrice'])/infoBook['askPrice']
            temp = (infoBook['askPrice']/(1+lastTakerFee)-infoBook['bidPrice']/(1-lastTakerFee))/(infoBook['askPrice']/(1+lastTakerFee))

            print("Market: %s,\tBID: %f,\tASK: %f,\tMargin: %f %%" % (market, infoBook['bidPrice'], infoBook['askPrice'], temp*100))

        #loop through ETH markets
        print("\nLooping through ETH markets...")
        for market in ethMarkets:
            book = polo.returnOrderBook(market)
            infoBook = utils.getBookInfo(book)
            #temp = (infoBook['askPrice']-infoBook['bidPrice'])/infoBook['askPrice']
            temp = (infoBook['askPrice']/(1+lastTakerFee)-infoBook['bidPrice']/(1-lastTakerFee))/(infoBook['askPrice']/(1+lastTakerFee))

            print("Market: %s,\tBID: %f,\tASK: %f,\tMargin: %f %%" % (market, infoBook['bidPrice'], infoBook['askPrice'], temp*100))
         '''

# time.sleep(1)
troll = 2



