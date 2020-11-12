import numpy as np
from scenredpy import Class_scenred
import h5py
import matplotlib.pyplot as plt
import copy
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale

# sr_instance = Class_scenred()
#
# sr_instance.import_data("test_data.h5")
# sr_instance.prepare_data()
# #
# # #sr_instance.scenario_reduction(dist_type="cityblock", fix_node=1, tol_node=np.linspace(1,24, 24)) #fix_prob_tol
# # sr_instance.scenario_reduction(dist_type="cityblock",fix_prob=1, tol_prob=np.linspace(0, 0.2, 24))
# #
# # sr_instance.draw_red_scenario()
# #
# # sr_instance.sort_result()
#
# Accessing original data
# f = h5py.File("test_data.h5", "r")
# name_list = list(f.keys())
# d = dict()
# for name in name_list:  # Stores keys on dictionary d
#     d[name] = f[name][()]
#
# i = 2
#
# #
# data = d
# dim_DAT = len(data)
# data_agg = np.hstack(data.values())
# dim_M, dim_N = data_agg.shape[0], int(data_agg.shape[1] /dim_DAT)
# print('Original data:')
# print(dim_M)
# print(dim_N)
# print(dim_DAT)
#
# # Testing how to access only a single data scenarios
# f = h5py.File("test_data.h5", "r")
# name_list = list(f.keys())
# # print(name_list)
# d = dict()
# d['param1'] = f['param1'][()]
#
# i = 2
#
# #
# data = d
# dim_DAT = len(data)
# data_agg = np.hstack(data.values())
# dim_M, dim_N = data_agg.shape[0], int(data_agg.shape[1] /dim_DAT)
# print('Your prototype:')
# print(dim_M)
# print(dim_N)
# print(dim_DAT)

# print(d['param1'][i])
# print(d['param2'][i])
# plt.plot(d['param1'][i])
# plt.plot(d['param1'][i+1], color='g')
# plt.show()

# Printing all dictionary contents
# fig = plt.figure()
# fig.add_subplot(2, 1, 1)
# for i in range(len(d['param1'])):
#     plt.plot(d['param1'][i], c=np.random.rand(3,))
#
# fig.add_subplot(2, 1, 2)
# for i in range(len(d['param2'])):
#     plt.plot(d['param2'][i], c=np.random.rand(3,))
# plt.show()

# Printing individual dictionary contents
# i = 2
# fig = plt.figure()
# fig.add_subplot(2, 1, 1)
# plt.plot(d['param1'][i])
# fig.add_subplot(2, 1, 2)
# plt.plot(d['param2'][i])
# plt.show()

# data_out, m_epo_res, m_epo_link, m_epo_prob = sr_instance.scenario_reduction(dist_type="cityblock",fix_prob=1, tol_prob=np.linspace(0, 0.2, 24))
#
# # Printing output contents
# for i in range(len(data_out['param1'])):
#     plt.plot(data_out['param1'][i], c=np.random.rand(3,))
#
# plt.show()
# i = 790
#
# print(len(data_out['param1']))
#
# for key in name_list:
#     fig = plt.figure(figsize=(8, 4))
#     ax = fig.add_subplot(111)
#     d_draw = data_out[key][m_epo_res[:, -1], :].transpose()
#     ax.plot(np.linspace(1, dim_N, dim_N), d_draw, color='gray', linewidth=0.5, alpha=0.5)
#     ax.set_xlim((1, dim_N))
#     ax.set_title(key)
#
# plt.show()

## Extracting functions to reduce scenarios directly
# Loading data
f = h5py.File("test_data.h5", "r")
def scenred(f):
    name_list = list(f.keys())
    d = dict()
    d['param1'] = f['param1'][()]
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

    return data_out

data_out = scenred(f)

for scenario in data_out['param1']:
    plt.plot(scenario)
plt.show()
