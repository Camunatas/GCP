# Loading and visualizing 2019 Spanish DAM prices
# Author: Pedro Luis Camuñas
#%% Importing libraries
import pandas as pd
import statsmodels.api as sm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as md
import datetime 

#%% Importing data
# Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])
# Setting interval
init = '2019-01-01 00:00:00'  # First hour to appear (YYYY-MM-DD)
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-12-31 23:00:00'  # Last hour to appear 
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

# Generating list with prices
Prices = []
for i in range(init_index, end_index):
    Prices.append(prices_df.iloc[i, 0])
	
# Generating list with dates
date_init = datetime.datetime.fromtimestamp(1546297200)
Dates = []
for i in range(init_index, end_index):
    Dates.append(datetime.datetime.fromtimestamp(1546297200 + 3600*i))

# Obtaining average price
Price_avg = sum(Prices) / len(Prices)

#%% Showing results
plt.plot(Dates, Prices, linewidth=0.25)
# plt.axhline(Price_avg, color='r', label='Average')
plt.xticks()
# plt.xlabel('Time')
plt.ylabel('Price (€/MWh)', fontsize=15)
plt.xticks(rotation=45)
plt.grid()
plt.legend()
plt.show()
