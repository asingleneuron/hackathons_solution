# USAID FORECASTING CHALLENGE

Converted layman timeseries problem to regression problem. After formulazing the regression problem applied the following two models. 


## EVALUATION METRIC: 

> Evaluation Metric for this competition is MASE

> MASE score has been calculated for each time-series after grouping data via product_code and site_code. After that calculating the average MASE. 



## MODEL_1 : LGBM

#### Validation MASE Score


| MASE | denominator |
|------|:------------|
| 0.9056 | only training_ts used |
| 0.7789 | training_ts + validation_ts used |



## MODEL_2 : CNN-LSTM

#### Validation MASE Score


| MASE | denominator |
|------|:------------|
| 0.8828 | only training_ts used |
| 0.7727 | training_ts + validation_ts used |


