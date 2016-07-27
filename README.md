# Model Comparisons

For this simple project I want to compare the accuracy of Random Forest, XGBoost, and Neural Network for Classification and Regression.  I will be using Forex data on a 5 minute tick.  The goal is to build a model using ~1000 ticks of data and then simulate live streaming data.  I will use the live streaming data to update the models and make a prediction on every new tick.  I will have another ~1000 ticks of data for my simulated live streaming.  

## Table of Contents
 - [Data Cleaning & EDA](https://github.com/gravity226/Model_Comparisons#data-cleaning--eda)
 - [Classification](https://github.com/gravity226/Model_Comparisons#random-forest-classification)

### Data Cleaning & EDA

See [data_cleaning_eda.py](https://github.com/gravity226/Model_Comparisons/blob/master/data_cleaning_eda.py) for the code.

Below are some of the initial metrics looked at from this dataset:

```
>> run data_cleaning_eda.py

Open ------------
Max 1.11648
Min 1.10145
STD 0.00303652916829

High ------------
Max 1.11861
Min 1.10233
STD 0.00303668942785

Low ------------
Max 1.11629
Min 1.10023
STD 0.00303981211749

Close ------------
Max 1.11648
Min 1.10146
STD 0.00303149011482
```

<img src="https://github.com/gravity226/Model_Comparisons/blob/master/imgs/line_chart.png" width="700" height="250" />

Classifying the data... If the closing price in a tick moved more than 0.0001 cents then it was classified as a 1 or a -1 (moved up or moved down).  If it didn't move more than 0.0001 cents then it was classified as a 0 (no significant change).
```
Making classes (y values in modeling)
1's 700
-1's 693
0's 655

First half classes (training set)
1's 355
-1's 367
0's 302
```
### Random Forest Classification

Starting out I wanted to see how the model would preform when given one data point at a time and then predicting the next data point.  Something interesting happens when I do this. From about tick 30 to tick 290 I am getting over %40 accuracy.  

<img src="https://github.com/gravity226/Model_Comparisons/blob/master/imgs/rf_validation_acc_from_zero.png" width="700" height="250" />
