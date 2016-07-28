from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML
from sklearn.ensemble import RandomForestClassifier

# my methods
from data_cleaning_eda import return_df

def from_zero():
    df = return_df()

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    # split = int(len(x) / 2)
    # x_train = x[:split]
    # y_train = y[:split]
    #
    # x_validate = x[split:]
    # y_validate = y[split:]
    #
    # model = RandomForestClassifier()
    # model.fit(x_train, y_train)

    num_correct = []
    weight = 50 # how much influence do you want the new ticks to have?

    model = RandomForestClassifier()

    model.fit(x[0], y[0])

    # simulate incoming timeseries data
    for tick in xrange(1, len(x)):
        if model.predict(x[tick]) == y[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        model = RandomForestClassifier(warm_start=True)

        model.fit(x[tick], y[tick])

    accuracy = sum(num_correct) / float(len(x))
        # print ''
        # print "accuracy:", accuracy

    acc_growth = [ sum(num_correct[:val])/float(val) for val in xrange(1, len(num_correct)+1) ]
    plt.plot(xrange(len(acc_growth)), acc_growth)
    plt.show()

def from_100(): # 0.5546875 accuracy
    df = return_df()

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    # split = int(len(x) / 2)
    # x_train = x[:split]
    # y_train = y[:split]
    #
    # x_validate = x[split:]
    # y_validate = y[split:]
    #
    # model = RandomForestClassifier()
    # model.fit(x_train, y_train)

    num_correct = []
    weight = 50 # how much influence do you want the new ticks to have?

    model = RandomForestClassifier()

    model.fit(x[0], y[0])

    # simulate incoming timeseries data
    for tick in xrange(1, len(x)):
        if model.predict(x[tick]) == y[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        model = RandomForestClassifier(warm_start=True)

        model.fit(x[tick], y[tick])

    accuracy = sum(num_correct) / float(len(x))
        # print ''
        # print "accuracy:", accuracy

    acc_growth = [ sum(num_correct[:val])/float(val) for val in xrange(1, len(num_correct)+1) ]
    plt.plot(xrange(len(acc_growth)), acc_growth)
    plt.show()

def from_dif():
    # from_zero()
    df = return_df()

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    # split = int(len(x) / 2)
    # x_train = x[:split]
    # y_train = y[:split]
    #
    # x_validate = x[split:]
    # y_validate = y[split:]
    #
    # model = RandomForestClassifier()
    # model.fit(x_train, y_train)

    num_correct = []
    weight = 50 # how much influence do you want the new ticks to have?

    model = RandomForestClassifier(warm_start=False)

    # set at 25 accuracy = 0.496786950074
    # set at 50 accuracy = 0.529029029029
    # set at 100 accuracy = 0.58932238193
    # set at 150 accuracy = 0.605374077977
    # set at 200 accuracy = 0.6130952380952381
    # set at 250 accuracy = 0.6201334816462737
    # set at 300 accuracy = 0.655606407323
    # set at 350 accuracy = 0.648409893993
    # set at 400 accuracy = 0.670509708738
    # set at 450 accuracy = 0.667709637046
    # set at 500 accuracy = 0.678294573643
    # set at 600 accuracy = 0.69544198895
    # set at 700 accuracy = 0.720326409496
    # set at 800 accuracy = 0.720352564103
    # set at 900 accuracy = 0.72212543554
    # set at 1000 accuracy = 0.749045801527
    # set at 1100 accuracy = 0.760548523207

    dist = 1100
    model.fit(x[:dist], y[:dist])

    # simulate incoming timeseries data
    for tick in xrange(dist, len(x)):
        if model.predict(x[tick]) == y[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        if tick > 1:
            model = RandomForestClassifier(warm_start=False)
            model.fit(x[tick-dist: tick], y[tick-dist: tick])
        else:
            model = RandomForestClassifier(warm_start=True)

            model.fit(x[tick], y[tick])

    accuracy = sum(num_correct) / float(len(num_correct))
    print ''
    print "accuracy:", accuracy

    acc_growth = [ sum(num_correct[:val])/float(val) for val in xrange(1, len(num_correct)+1) ]
    plt.plot(xrange(len(acc_growth)), acc_growth)
    plt.show()
    
if __name__ == '__main__':
    # from_zero()
    df = return_df()

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    # split = int(len(x) / 2)
    # x_train = x[:split]
    # y_train = y[:split]
    #
    # x_validate = x[split:]
    # y_validate = y[split:]
    #
    # model = RandomForestClassifier()
    # model.fit(x_train, y_train)

    num_correct = []
    weight = 50 # how much influence do you want the new ticks to have?

    model = RandomForestClassifier(warm_start=False)

    # set at 25 accuracy = 0.496786950074
    # set at 50 accuracy = 0.529029029029
    # set at 100 accuracy = 0.58932238193
    # set at 150 accuracy = 0.605374077977
    # set at 200 accuracy = 0.6130952380952381
    # set at 250 accuracy = 0.6201334816462737
    # set at 300 accuracy = 0.655606407323
    # set at 350 accuracy = 0.648409893993
    # set at 400 accuracy = 0.670509708738
    # set at 450 accuracy = 0.667709637046
    # set at 500 accuracy = 0.678294573643
    # set at 600 accuracy = 0.69544198895
    # set at 700 accuracy = 0.720326409496
    # set at 800 accuracy = 0.720352564103
    # set at 900 accuracy = 0.72212543554
    # set at 1000 accuracy = 0.749045801527
    # set at 1100 accuracy = 0.760548523207

    dist = 1100
    model.fit(x[:dist], y[:dist])

    # simulate incoming timeseries data
    for tick in xrange(dist, len(x)):
        if model.predict(x[tick]) == y[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        if tick > 1:
            model = RandomForestClassifier(warm_start=False)
            model.fit(x[tick-dist: tick], y[tick-dist: tick])
        else:
            model = RandomForestClassifier(warm_start=True)

            model.fit(x[tick], y[tick])

    accuracy = sum(num_correct) / float(len(num_correct))
    print ''
    print "accuracy:", accuracy

    acc_growth = [ sum(num_correct[:val])/float(val) for val in xrange(1, len(num_correct)+1) ]
    plt.plot(xrange(len(acc_growth)), acc_growth)
    plt.show()

























#
