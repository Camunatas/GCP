# Script for loading electricity price data from csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
# Setting interval
init = '2019-01-01 00:00:00'                # First hour to appear
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-01-01 23:00:00'                # Last hour to appear
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

# Generating list with prices and hours
prices = []
hours = []
for i in range(init_index, end_index):
    prices.append(prices_df.iloc[i, 0])
    hours.append(prices_df.iloc[i, 1])

plt.plot(prices)
plt.show()
# print(prices_df.iloc[1, 0])
# print(prices_df.iloc[init_index, 0])
