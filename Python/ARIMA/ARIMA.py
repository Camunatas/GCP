# Forecast with ARIMA model of electricity price
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

#%% Loading data
# Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
# Setting interval
init = '2019-01-01 00:00:00'  # First hour to appear
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-03-21 23:00:00'  # Last hour to appear
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

# Generating list with prices and hours
prices = []
hours = []
for i in range(init_index, end_index):
    prices.append(prices_df.iloc[i, 0])
    hours.append(prices_df.iloc[i, 1])

# Creating datasets
train = 50
test = 1
prices_train = list(prices[0:24 * train])
prices_test = list(prices[24 * train:24 * (train + test)])

#%% Setting ARIMA model
# Creating arima model
model_order = (2, 0, 0)
model_seasonal_order = (2, 1, 1, 24)

model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order, seasonal_order=model_seasonal_order)
# Fitting model
model_fit = model.fit(disp=0)

# Printing day-ahead forecast
prices_pred = model_fit.forecast(steps=24)
print(prices_test)
print(prices_pred)

#%% Plotting results
# Plotting day ahead forecast & real values comparison
plt.plot(prices_test, label='Real')
plt.plot(prices_pred, color='red', label='Prediction')
plt.legend()
plt.xlabel("Time (h)")
plt.ylabel("Price (€/MWh)")
plt.show()