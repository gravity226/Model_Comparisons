from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ML
from sklearn.ensemble import RandomForestClassifier

# my methods
from data_cleaning_eda import return_df

if __name__ == '__main__':

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

























#
