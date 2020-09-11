# USAID FORECASTING CHALLENGE

Converted layman timeseries problem to regression problem. After formulazing the regression problem applied the following two models. 

<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>

## EVALUATION METRIC: 

> Evaluation Metric for this competition is MASE

> MASE score has been calculated for each time-series after grouping data via product_code and site_code. After that calculating the average MASE. 

<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>


## MODEL_1 : LGBM

#### Validation MASE Score


| MASE | denominator |
|------|:------------|
| 0.9056 | only training_ts used |
| 0.7789 | training_ts + validation_ts used |

#### Validation Plot

Following validation plot represents the following things:
1. Training and Validation time-period
2. Training, Validation and Predicted time-series ( Date as x-axis)
3. Training, Validation and Predicted time-series ( index as x-axis)
4. Only Validation and Predicted time-series 
5. Resudial error plot of Validation and Predicted time-series

![LGB Validaiton Plot](https://github.com/asingleneuron/hackathons_solution/blob/master/usaid_forecasting_solution/usaid_validation_plots/lgb_validation_ts_plot.png)


Following forecasted plot represents the following things:
1. Training and Forecasting time-period
2. Training, Forecasted time-series ( Date as x-axis)
3. Training, Forecasted time-series ( index as x-axis)
4. Only Forecasted time-series 


![LGB Forecasting Plot](https://github.com/asingleneuron/hackathons_solution/blob/master/usaid_forecasting_solution/usaid_validation_plots/lgb_forecasted_ts_plot.png)

<p>&nbsp;</p>
<p>&nbsp;</p>
<p>&nbsp;</p>


## MODEL_2 : CNN-LSTM

#### Validation MASE Score


| MASE | denominator |
|------|:------------|
| 0.8828 | only training_ts used |
| 0.7727 | training_ts + validation_ts used |




#### Validation Plot

Following validation plot represents the following things:
1. Training and Validation time-period
2. Training, Validation and Predicted time-series ( Date as x-axis)
3. Training, Validation and Predicted time-series ( index as x-axis)
4. Only Validation and Predicted time-series 
5. Resudial error plot of Validation and Predicted time-series

![CNN-LSTM Validaiton Plot](https://github.com/asingleneuron/hackathons_solution/blob/master/usaid_forecasting_solution/usaid_validation_plots/cnn_lstm_validation_ts_plot.png)


Following forecasted plot represents the following things:
1. Training and Forecasting time-period
2. Training, Forecasted time-series ( Date as x-axis)
3. Training, Forecasted time-series ( index as x-axis)
4. Only Forecasted time-series 


![CNN-LSTM Forecasting Plot](https://github.com/asingleneuron/hackathons_solution/blob/master/usaid_forecasting_solution/usaid_validation_plots/cnn_lstm_forecasted_ts_plot.png)




