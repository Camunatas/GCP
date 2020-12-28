# This script automatizes the process to obtain the best ARIMA model for a series of data, it prints the results for t
# the optimal ARIMA model. It will choose the one with the lower Akaike Information Critera (AIC)

#%% Importing libraries
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
import numpy as np

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

#%% Model order grid search
# model = pm.auto_arima(prices, start_p=0, start_q=0,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=3, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=True,    # Seasonality
#                       start_P=0,
#                       D=0,
#                       trace=True,
#                       error_action='ignore',
#                       suppress_warnings=True,
#                       stepwise=True)
#
# print(model.summary())
model = pm.auto_arima(prices_train, start_p=0, start_q=0,
					    m=24,start_P=0, seasonal=True,
						trace=True, error_action='ignore',
						   suppress_warnings=True,
                           stepwise=True)
print(model.summary())