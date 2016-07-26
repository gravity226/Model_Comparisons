# Model Comparisons

For this simple project I want to compare the accuracy of Random Forest, XGBoost, and Neural Network for Classification and Regression.  I will be using Forex data on a 5 minute tick.  The goal is to build a model using ~1000 ticks of data and then simulate live streaming data.  I will use the live streaming data to update the models and make a prediction on every new tick.  I will have another ~1000 ticks of data for my simulated live streaming.  

## Table of Contents
 - [Data Cleaning & EDA](https://github.com/gravity226/Model_Comparisons#data-cleaning--eda)

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
