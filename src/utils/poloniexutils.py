__author__ = "Michel Tulane"
#File created 27-DEC-2017


def getBookInfo(orderBook):

    bookInfo = {'bidPrice': float(orderBook['bids'][0][0]), 
                'askPrice': float(orderBook['asks'][0][0])}
    return bookInfo







