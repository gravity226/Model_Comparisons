from __future__ import division
import pandas as pd
import numpy as np

def rf_model(df):
    from sklearn.ensemble import RandomForestClassifier

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    split = int(len(x) / 2)
    x_train = x[:split]
    y_train = y[:split]

    x_validate = x[split:]
    y_validate = y[split:]

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    num_correct = []
    weight = 50 # how much influence do you want the new ticks to have?

    # simulate incoming timeseries data
    for tick in xrange(len(x_validate)):
        if model.predict(x_validate[tick]) == y_validate[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        model = RandomForestClassifier(warm_start=True)

        new_x = [ x_validate[tick] for val in xrange(weight) ]
        new_y = [ y_validate[tick] for val in xrange(weight) ]
        model.fit(new_x, new_y)

    accuracy = sum(num_correct) / float(len(x_validate))
        # print ''
        # print "accuracy:", accuracy

    import matplotlib.pyplot as plt
    acc_growth = [ sum(num_correct[:val])/float(val) for val in xrange(1, len(num_correct)+1) ]
    plt.plot(xrange(len(acc_growth)), acc_growth)
    plt.show()

    return accuracy

def xgboost_model(df):
    # import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    split = int(len(x) / 2)
    x_train = x[:split]
    y_train = y[:split]

    x_validate = x[split:]
    y_validate = y[split:]

    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    num_correct = []

    # simulate incoming timeseries data
    for tick in xrange(len(x_validate)):
        if model.predict(x_validate[tick]) == y_validate[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        model.fit(x_validate[tick], y_validate[tick])

    accuracy = sum(num_correct) / float(len(x_validate))
    # print ''
    # print "accuracy:", accuracy

    # import matplotlib.pyplot as plt
    # plt.plot(xrange(len(num_correct)), num_correct)
    return accuracy

def boosted_model(df):
    from sklearn.ensemble import AdaBoostClassifier

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    split = int(len(x) / 2)
    x_train = x[:split]
    y_train = y[:split]

    x_validate = x[split:]
    y_validate = y[split:]

    model = AdaBoostClassifier()
    model.fit(x_train, y_train)

    num_correct = []

    # simulate incoming timeseries data
    for tick in xrange(len(x_validate)):
        if model.predict(x_validate[tick]) == y_validate[tick]:
            num_correct.append(1)
        else:
            num_correct.append(0)

        # update model with new data
        model.fit(x_validate[tick], [y_validate[tick]])

    accuracy = sum(num_correct) / float(len(x_validate))
    # print ''
    # print "accuracy:", accuracy

    # import matplotlib.pyplot as plt
    # plt.plot(xrange(len(num_correct)), num_correct)
    return accuracy

def nn_model(df):
    import tensorflow as tf
    import math
    from tqdm import tqdm

    x = df[['open','high','low','close']].values
    y = df['classes'].values

    train = np.array([ [[t / x.max() for t in tick]] for tick in x ])
    labels = y

    def to_onehot(labels,nclasses = 3):
        '''
        Convert labels to "one-hot" format.

        >>> a = [0,1,2,3]
        >>> to_onehot(a,5)
        array([[ 1.,  0.,  0.,  0.,  0.],
               [ 0.,  1.,  0.,  0.,  0.],
               [ 0.,  0.,  1.,  0.,  0.],
               [ 0.,  0.,  0.,  1.,  0.]])
        '''
        outlabels = np.zeros((len(labels),nclasses))
        for i,l in enumerate(labels):
            outlabels[i,l] = 1
        return outlabels

    onehot = to_onehot(labels, 3)

    indices = np.random.permutation(train.shape[0])
    valid_cnt = int(len(x) / 2)

    test, train = train[valid_cnt:], train[:valid_cnt] # train[test_idx,:], train[training_idx,:]
    onehot_test, onehot_train = onehot[valid_cnt:], onehot[:valid_cnt]

    sess = tf.InteractiveSession()

    # These will be inputs
    ## Input pixels, flattened
    x = tf.placeholder("float", [None, 4])
    ## Known labels
    y_ = tf.placeholder("float", [None,3])


    # Hidden layer 1
    num_hidden1 = 128
    img_shape = train.shape[1] * train.shape[2]
    y_shape = 3

    W1 = tf.Variable(tf.truncated_normal([img_shape,num_hidden1],
                                   stddev=1./math.sqrt(img_shape)))

    b1 = tf.Variable(tf.constant(0.1,shape=[num_hidden1]))
    h1 = tf.sigmoid(tf.matmul(x,W1) + b1)


    # Hidden Layer 2
    num_hidden2 = 128
    W2 = tf.Variable(tf.truncated_normal([num_hidden1,
                num_hidden2],stddev=2./math.sqrt(num_hidden1)))

    b2 = tf.Variable(tf.constant(0.2,shape=[num_hidden2]))
    h2 = tf.sigmoid(tf.matmul(h1,W2) + b2)


    # Hidden Layer 3
    num_hidden4 = 64
    W4 = tf.Variable(tf.truncated_normal([num_hidden2,
                num_hidden4],stddev=2./math.sqrt(num_hidden2)))

    b4 = tf.Variable(tf.constant(0.2,shape=[num_hidden4]))
    h4 = tf.sigmoid(tf.matmul(h2,W4) + b4)


    # Hidden Layer 4
    num_hidden5 = 64
    W5 = tf.Variable(tf.truncated_normal([num_hidden4,
                num_hidden5],stddev=2./math.sqrt(num_hidden4)))

    b5 = tf.Variable(tf.constant(0.2,shape=[num_hidden5]))
    h5 = tf.sigmoid(tf.matmul(h4,W5) + b5)


    # Hidden Layer 5
    num_hidden3 = 32
    new_W = tf.Variable(tf.truncated_normal([num_hidden4,
                num_hidden3],stddev=2./math.sqrt(num_hidden4)))

    new_b = tf.Variable(tf.constant(0.2,shape=[num_hidden3]))
    new_h = tf.sigmoid(tf.matmul(h4,new_W) + new_b)


    # Output Layer
    W3 = tf.Variable(tf.truncated_normal([num_hidden3, y_shape],
                                       stddev=1./math.sqrt(y_shape)))
    b3 = tf.Variable(tf.constant(0.1,shape=[y_shape]))


    # Just initialize
    sess.run(tf.initialize_all_variables())

    # Define model
    y = tf.nn.softmax(tf.matmul(new_h,W3) + b3)

    ### End model specification, begin training code


    # Climb on cross-entropy
    cross_entropy = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(y + 1e-50, y_))

    # How we train
    train_step = tf.train.GradientDescentOptimizer(0.009).minimize(cross_entropy)

    # Define accuracy
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Actually train
    epochs = 1000 # 25000
    train_acc = np.zeros(epochs//10)
    test_acc = np.zeros(epochs//10)
    for i in tqdm(range(epochs), ascii=True):
        if i % 10 == 0:  # Record summary data, and the accuracy
            # Check accuracy on train set
            A = accuracy.eval(feed_dict={x: train.reshape([-1,4]), y_: onehot_train})
            train_acc[i//10] = A

            # And now the validation set
            A = accuracy.eval(feed_dict={x: test.reshape([-1,4]), y_: onehot_test})
            test_acc[i//10] = A
        train_step.run(feed_dict={x: train.reshape([-1,4]), y_: onehot_train})

    # print "Accuracy:", A

    return A


if __name__ == '__main__':
    from data_cleaning_eda import return_df
    df = return_df()

    rf_acc =  rf_model(df) # 0.3671875 (before adding weight to new ticks)
    ada_acc = 0.3681640625 # boosted_model(df)
    nn_acc = 0.336914 # nn_model(df) # no time component

    print "Random Forest Accuracy:", rf_acc     # 0.3671875 # 0.3681640625 after adding weight
    print "Ada Boosted Accuracy:  ", ada_acc    # 0.3681640625
    print "Dense Net Accuracy:    ", nn_acc     # 0.336914
























#
