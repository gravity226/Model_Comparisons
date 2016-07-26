import numpy as np
import pandas as pd


def get_data():
    return pd.read_csv('data/EURUSDecn5.csv')

    # df.columns
    # [u'date', u'time', u'open', u'high', u'low', u'close', u'something']

    # df.info
    # <class 'pandas.core.frame.DataFrame'>
    # Int64Index: 2048 entries, 0 to 2047
    # Data columns (total 7 columns):
    # date         2048 non-null object
    # time         2048 non-null object
    # open         2048 non-null float64
    # high         2048 non-null float64
    # low          2048 non-null float64
    # close        2048 non-null float64
    # something    2048 non-null int64
    # dtypes: float64(4), int64(1), object(2)
    # memory usage: 128.0+ KB

def get_metrics(df):
    # [u'date', u'time', u'open', u'high', u'low', u'close', u'something']
    print ''
    print "Open ------------"
    print "Max", df['open'].max()
    print "Min", df['open'].min()
    print "STD", df['open'].std()

    print ''
    print "High ------------"
    print "Max", df['high'].max()
    print "Min", df['high'].min()
    print "STD", df['high'].std()

    print ''
    print "Low ------------"
    print "Max", df['low'].max()
    print "Min", df['low'].min()
    print "STD", df['low'].std()

    print ''
    print "Close ------------"
    print "Max", df['close'].max()
    print "Min", df['close'].min()
    print "STD", df['close'].std()

def plot_timeseries(df):
    import matplotlib.pyplot as plt

    plt.plot(xrange(len(df['close'])), df['close'])
    plt.savefig('line_chart.png')
    plt.show()

if __name__ == '__main__':
    df = get_data()
    # get_metrics(df)
    # plot_timeseries(df)


























#
