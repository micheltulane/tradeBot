__author__ = "Michel Tulane"
#File created 13-OCT-2018

import pandas as pd

def read_cryptodatadownload_csv(filename):
    """
    This functions reads a cryptocurrency pair exchange data (csv file) obtained from www.cryptodatadownload.com
    :param filename: Path of the csv file
    :return: dataframe re-arranged in chronological order (lower indexes are older values)
    """

    df = pd.read_csv(filename, skiprows=1)
    df = df.sort_index(axis=0, ascending=False)
    df = df.reset_index(drop=True)

    return df


