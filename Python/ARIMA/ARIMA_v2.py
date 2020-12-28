# Forecast with ARIMA model of electricity price
#%% Importing libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
import datetime 
#%% Importing data
# Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
# Setting interval
init = '2019-01-01 00:00:00'  # First hour to appear
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-12-31 23:00:00'  # Last hour to appear
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

# Generating list with prices and hours
prices = []
hours = []
for i in range(init_index, end_index):
    prices.append(prices_df.iloc[i, 0])
    hours.append(prices_df.iloc[i, 1])

# Creating datasets
train = 100
test = 1
prices_train = list(prices[0:24 * train])
prices_test = list(prices[24 * train:24 * (train + test)])

# Generating list with dates
date_init = datetime.datetime.fromtimestamp(1546297200)
Dates = []
for i in range(end_index, end_index+24*test):
    Dates.append(datetime.datetime.fromtimestamp(1546297200 + 3600*i).hour + 
				 datetime.datetime.fromtimestamp(1546297200 + 3600*i).minute)

#%% Analizing data
# Autocorrelation plot
# autocorrelation_plot(prices_train)
# sm.graphics.tsa.plot_acf(prices_train, lags=40)
#%% Setting SARIMA model
model_order = (1, 2, 1)
model_seasonal_order = (1, 1, 1, 24)

model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order, seasonal_order=model_seasonal_order)
# model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order)
# Fitting model
model_fit = model.fit(disp=0)

#%% Presenting results
# Printing day-ahead forecast
prices_pred = model_fit.forecast(steps=24)
print(prices_test)
print(prices_pred)

# Creating dates label
dates_label = []
for i in range(24):
	dates_label.append('{}:00'.format(i))
	
# Plotting day ahead forecast & real values comparison
ax = plt.figure()
plt.plot(Dates, prices_pred, color='red', label='Prediction')
plt.plot(Dates, prices_test, label='Real')
# plt.bar(Dates, prices_pred, color='red', label='Prediction', width = 0.5, align='center', edgecolor='black')
# plt.bar(Dates, prices_test, label='Real', width = 0.5, align='edge', edgecolor='black')
plt.xticks(np.arange(0, 24, 1), dates_label, rotation=45)
plt.ylim(bottom=45)
plt.ylim(top=62)
plt.legend()
plt.xlabel("Hour")
plt.ylabel("Price (€/MWh)")
plt.show()