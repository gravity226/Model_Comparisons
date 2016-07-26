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
    plt.savefig('imgs/line_chart.png')
    plt.show()

def classing(col, param=0.0001):
    classes = []
    for row in xrange(1, len(col)):
        # how much of a change to you have here?
        if col[row] > (col[row - 1] - param) and col[row] < (col[row - 1] + param):
            classes.append(0)
        elif col[row] > col[row - 1]:
            classes.append(1)
        elif col[row] < col[row - 1]:
            classes.append(-1)
    return classes

def return_df():
    df = get_data()
    df['classes'] = [0] + classing(df['close'].values)
    return df
################################################################################

if __name__ == '__main__':
    df = get_data()
    get_metrics(df)
    # plot_timeseries(df)
    df['classes'] = [0] + classing(df['close'].values)

    print ''
    print "Making classes (y values in modeling)"
    q = list(df.classes.values)
    print "1's", q.count(1)
    print "-1's", q.count(-1)
    print "0's", q.count(0)
    '''
    1's 700
    -1's 693
    0's 655
    '''

    print ''
    print "First half classes (training set)"
    q = list(df.classes.values)
    print "1's", q[:int(len(q)/2)].count(1)
    print "-1's", q[:int(len(q)/2)].count(-1)
    print "0's", q[:int(len(q)/2)].count(0)
    '''
    1's 355
    -1's 367
    0's 302
    '''



















#
