# Script for generating prediction scenarios of an ARIMA model
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import random as rand

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

# Creating arima model
model_order = (1, 1, 2)
model_seasonal_order = (0, 1, 1, 24)

model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order, seasonal_order=model_seasonal_order)
model_fit = model.fit(disp=0)

# Model diagnosis
# model_fit.plot_diagnostics(figsize=(15, 12))
# plt.show()

# Getting confidence intervals for next 24 out-of-sample predictions
conf_ins = model_fit.get_forecast(24).summary_frame()
print(conf_ins)


# Obtaining KDEs for each hour
def hourly_kde(h):
    samples = 999
    s = [np.random.normal(conf_ins.iloc[h, 0], (conf_ins.iloc[h, 3] - conf_ins.iloc[h, 2])/3) for i in range(samples)]
    kde = stats.gaussian_kde(s)
    kde.set_bandwidth(bw_method='silverman')

    return kde


# Storing KDEs in dictionary
kdes = {}
for i in range(24):
    print("Hour {}".format(i))
    kdes["{}".format(i)] = hourly_kde(i)


# Generating scenario
def scenario_generator():
    scnr = []
    for i in range(24):
        pred_scn_h = kdes["{}".format(i)].resample(1)[0][0]
        scnr.append(pred_scn_h)

    return scnr


# Plotting scenarios
n_scenarios = 100  # Number of scenarios
for i in range(n_scenarios):
    plt.plot(scenario_generator(), c=np.random.rand(3,), label='Scenario')
    plt.xlabel("Hour")
    plt.ylabel("Price (€/MWh)")
plt.show()
