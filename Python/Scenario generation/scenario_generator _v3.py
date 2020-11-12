# Script for generating prediction scenarios of an ARIMA model by using normal distributions

#%% Importing libraries
import pandas as pd
import numpy as np
# import pmdarima as pm
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
import h5py
import copy
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale

#%% Loading csv
fields = ["Price", "Hour"]
prices_df = pd.read_csv('Prices_2019.csv', sep=';', usecols=fields, parse_dates=[1])

#%% Setting interval
init = '2019-01-01 00:00:00'  # First hour to appear
init_index = np.where(prices_df["Hour"] == init)[0][0]
end = '2019-12-31 23:00:00'  # Last hour to appear
end_index = np.where(prices_df["Hour"] == end)[0][0] + 1

#%% Creating training and test data
prices = []
hours = []
for i in range(init_index, end_index):
    prices.append(prices_df.iloc[i, 0])
    hours.append(prices_df.iloc[i, 1])

train = 150
test = 1
prices_train = list(prices[0:24 * train])
prices_test = list(prices[24 * train:24 * (train + test)])

#%% Creating arima model (SEASONAL)
model_order = (2, 0, 0)
model_seasonal_order = (2, 1, 1, 24)
model_order = (8, 0, 6)
model = sm.tsa.statespace.SARIMAX(prices_train, order=model_order)
# model = pm.auto_arima(prices_train, start_p=0, start_q=0,
#                            max_p=10, max_q=10, m=24,
#                            start_P=0, seasonal=True,
#                            d=0, D=1, trace=True,
#                            error_action='ignore',
#                            suppress_warnings=True,
#                            stepwise=True)
model_fit = model.fit(disp=0)


#%% Generating scenarios
n_scenarios = 100  # Number of scenarios
sim = model_fit.simulate(nsimulations = n_scenarios, repetitions=24, anchor='end')
sim = np.reshape(sim,[n_scenarios,24])

# Plotting scenarios
for i in range(n_scenarios):
    plt.plot(sim[i], c=np.random.rand(3,), label='Scenario')
    # plt.plot(prices_test, linestyle='dotted', color='red')
    plt.xlabel("Hour")
    plt.ylabel("Price (€/MWh)")
plt.show()

#%% Reducing scenarios
f = h5py.File("test_data.h5", "r")
def scenred(scenarios):
    d = dict()
    d['param1'] = scenarios[()]
    data = d
    # Generating auxiliary data variables
    dim_DAT = len(data)
    data_agg = np.hstack(data.values())
    dim_M, dim_N = data_agg.shape[0], int(data_agg.shape[1] / dim_DAT)
    # Reducing scenarios
    dist_type = "cityblock"
    fix_prob = 1
    tol_prob = np.linspace(0, 0.2, 24)
    tol_node = np.full(dim_N, 1)  # by default, no of nodes must be larger than 1.

    # Selection:
    sel = dict()
    sel["fix_prob"] = fix_prob
    sel["fix_node"] = fix_prob

    # PRE-DEFINED
    nodes_left = data_agg.shape[0]

    # prob matrix
    m_epo_prob = np.full([dim_M, dim_N], 1)

    # distance matrix
    # get distance
    data_normalized = scale(data_agg, axis=0)  # z-score

    m_dist = cdist(data_normalized, data_normalized, dist_type, p=2)  # distance #'cityblock' 'minkowski', p=2
    m_dist_aug = m_dist + np.eye(np.size(data_agg, 0)) * (1 + np.max(m_dist))

    # distance probability matrixs
    m_dist_prob = np.full(m_dist_aug.shape, True, dtype=bool)

    # residue matrix
    m_epo_res = np.full([dim_M, dim_N], True, dtype=bool)

    # linkage matrix in nodes:
    m_link = np.full(m_dist_aug.shape, 0)
    # linkage matrix in epoch:
    m_epo_link = -np.full([dim_M, dim_N], 1)

    data_out = copy.deepcopy(data)

    ## START ITERATION:
    for epo in np.flip(np.arange(dim_N), axis=0):  # backward
        dp_limit = 1 * tol_prob[epo] * np.min([i for i in np.sum(m_dist * m_dist_prob, axis=0) if i > 0])
        dp = 0

        counter = True
        while counter:
            m_dist_aug[~m_epo_res[:, epo], :] = np.Inf
            m_dist_aug[:, ~m_epo_res[:, epo]] = np.Inf

            c_min_M = np.min(m_dist_aug, axis=1)
            c_min_M_idx = np.argmin(m_dist_aug, axis=1)

            # min z
            arr_z = c_min_M * m_epo_prob[:, epo]
            for i in np.arange(arr_z.shape[0]):
                if np.isnan(arr_z[i]) == True or arr_z[i] == 0:
                    arr_z[i] = np.Inf

            del_val = np.min(c_min_M)
            del_idx = np.argmin(c_min_M)

            aug_idx = c_min_M_idx[del_idx]

            dp += del_val

            # use mode selection:

            if dp <= dp_limit and nodes_left > tol_node[epo]:
                # transfer probability:
                m_epo_prob[aug_idx, epo] += m_epo_prob[del_idx, epo]

                # delete probability with related index:
                m_dist_prob[del_idx, :] = False
                m_dist_prob[:, del_idx] = False

                # delete probability in epoch:
                m_epo_res[del_idx, epo] = False
                m_epo_prob[del_idx, epo] = 0

                # calculate nodes left
                nodes_left = np.sum(m_epo_prob[:, epo] > 0)

                # recunstruct link matrix in nodes:
                m_link[aug_idx, del_idx] = 1
                # inherit deleted index
                m_link[aug_idx, m_link[del_idx, :] == 1] = 1

                # recunstruct link matrix in epoch:
                m_epo_link[del_idx, epo] = aug_idx
                # check if this del_idx is a formal aug_idx:
                m_epo_link[np.where(m_epo_link[:, epo] == del_idx), epo] = aug_idx

                to_merge_idx = np.where(m_link[del_idx, :])
                # remove data from deleted scenario:
                for ele_data in data_out.values():
                    ele_data[del_idx, 0:epo + 1] = ele_data[aug_idx, 0:epo + 1]
                    # inherit from deleted index, if it has information from other samples:
                    for ele in to_merge_idx:
                        ele_data[ele, 0:epo + 1] = ele_data[aug_idx, 0:epo + 1]
            else:
                # close current iteration
                counter = False
                dp -= del_val
        # end while

        if epo > 0:
            # Update available scenarios in the previous epoch:
            m_epo_res[:, epo - 1] = m_epo_res[:, epo]
            m_epo_prob[:, epo - 1] = m_epo_prob[:, epo]

        print("epo={1}: {0} nodes left ".format(nodes_left, epo))

    ## END ITERATION
    data_out = data_out
    m_epo_res = m_epo_res
    m_epo_link = m_epo_link
    m_epo_prob = m_epo_prob

    return data_out, d

data_out,d = scenred(sim)

for scenario in data_out['param1']:
    plt.plot(scenario)
plt.show()
