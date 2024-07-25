import itertools
import pandas as pd
import numpy as np
from random import gauss

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess

from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error

import os
from IPython.display import Image

import matplotlib.pyplot as plt
import warnings
import itertools
import statsmodels.api as sm
from random import gauss
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from pmdarima.arima import auto_arima
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima

from prophet import Prophet 

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 

warnings.simplefilter(action='ignore', category= FutureWarning)

#ar1 = np.array([1, -0.9])
#ma1 = np.array([1])
#AR_object1 = ArmaProcess(ar1, ma1)
#simulated_data_1 = AR_object1.generate_sample(nsample=1000)
#plt.plot(simulated_data_1)
#plt.show()

#plot_acf(simulated_data_1, lags = 10)
#plt.show()

# load the data
train = pd.read_csv('train.csv', parse_dates=['date'], index_col='date')

df = pd.concat([train],sort=True)

# we subset to one item x store combination
buf = df[(df.item==1)&(df.store==1)].copy()
buf.head(10)

# what do the components look like? 
#decomposition = seasonal_decompose(buf.sales.dropna(), period=365)
#figure = decomposition.plot()
#plt.show()

buf = buf.sort_index()



tr_start,tr_end = '2014-01-01','2017-09-30'
te_start,te_end = '2017-10-01','2017-12-31'
x0 = buf['sales'][tr_start:tr_end].dropna()
x1 = buf['sales'][te_start:te_end].dropna()

x1 = x1[~x1.index.duplicated(keep='first')]

# examine autocorrelation
plot_acf(x0, lags = 12); print()
plot_pacf(x0, lags = 12); print()

model_autoARIMA = auto_arima(x0, start_p=7, start_q=7 ,
                      test='adf',       
                      max_p= 7, max_q=7, 
                      m= 7,              
                      d= 1,
                      seasonal=True,   
                      start_P=1, 
                      D=1, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

#print(model_autoARIMA.summary())

#model_autoARIMA.plot_diagnostics()
#plt.show()



# Forecast (ensure n_periods matches length of x1)
forecast = model_autoARIMA.predict(n_periods=len(x1))


# Ensure 'pred' is a pandas Series with the same index as 'x1'
#pred_series = pd.Series(pred, index=x1.index)

# Convert forecast to Series with x1 index
pred_series = pd.Series(forecast, index=x1.index)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x1.index, x1, label='Actual', color='blue')
plt.plot(pred_series.index, pred_series, label='Forecast', color='red')
plt.legend()
plt.title('Actual vs Forecast')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()



# Align 'pred' to 'x1' index (assuming predictions are for the future)
#pred_aligned = pred_series.reindex(x1.index)


#pd.DataFrame({'test': x1, 'pred': pred_series}).plot()
#plt.show()

#Checking for duplicates

#print(x1.index.duplicated().any())

#duplicates = x1.index[x1.index.duplicated()]
#print(duplicates)


