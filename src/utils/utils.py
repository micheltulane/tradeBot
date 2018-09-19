#File created 27-DEC-2017
#Author: Michel Tulane

#using Python API wrapper for Poloniex: https://github.com/s4w3d0ff/python-poloniex/archive/v0.4.7.zip

def getBookInfo(orderBook):

    bookInfo = {'bidPrice': float(orderBook['bids'][0][0]), 
                'askPrice': float(orderBook['asks'][0][0])}
    return bookInfo







