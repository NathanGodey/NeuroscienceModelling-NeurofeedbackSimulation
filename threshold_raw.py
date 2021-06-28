import numpy as np
import scipy.stats as stats
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import random

rng = np.random.default_rng()

def avg(arr):
    return arr
    return np.cumsum(arr)/np.arange(1,len(arr)+1)

UAF_list_tar0 = pickle.load(open("saved_data/UAF_list_baseline_5mn.npy", "rb"))
UAF_list_tar1= pickle.load(open("saved_data/UAF_list_baseline+1_5mn.npy", "rb"))

np.random.shuffle(UAF_list_tar0) #allow sampling by incrementation
np.random.shuffle(UAF_list_tar1)

min_T = np.min(UAF_list_tar0)
max_T = 160

target_id = 3
K = 50
n_thresh = 30
learning_rate = 0.1
T = 10000

# target_weights : one line  = one threshold value, one column = weight value at time t
target_weights = np.zeros((n_thresh, T))
percent_learners = [0 for i in range(n_thresh)]

for i_thresh, threshold in enumerate(tqdm(np.linspace(min_T, max_T, n_thresh))):
    for k in range(K):
        start = time.time()
        i_dist_0 = 0
        i_dist_1 = 0
        msn_weights = np.ones(1000)
        for t in range(T):
            norm_weights = np.maximum(msn_weights, 1e-15)
            S = norm_weights.sum()
            if S > 0:
                norm_weights = norm_weights / S
            active_units = rng.choice(np.arange(1000), 10, p=norm_weights, replace=False)
            target_in_active = target_id in active_units

            if target_in_active:
                UAF = UAF_list_tar1[i_dist_1]
                i_dist_1 = (i_dist_1 + 1) % len(UAF_list_tar1)
            else:
                UAF = UAF_list_tar0[i_dist_0]
                i_dist_0 = (i_dist_0 + 1) % len(UAF_list_tar0)

            if UAF > threshold:
                msn_weights[active_units] += learning_rate
            else:
                msn_weights[active_units] -= learning_rate

            target_weights[i_thresh, t] += norm_weights[target_id]/K
        percent_learners[i_thresh] += int(msn_weights[target_id] > (1 + 3*learning_rate))/K


    plt.plot(avg(target_weights[i_thresh]), color=plt.get_cmap('plasma')(i_thresh/n_thresh))
plt.xlabel('Time (ms)')
plt.ylabel('Target unit weight')
plt.show()
plt.plot(np.linspace(min_T, max_T, n_thresh), [target_weights[i_thresh, -1] for i_thresh in range(n_thresh)])
plt.xlabel('UAF threshold')
plt.ylabel('Final weight value')
plt.show()
plt.plot(np.linspace(min_T, max_T, n_thresh), percent_learners)
plt.xlabel('UAF threshold')
plt.ylabel('Rate of successful learning sessions')
plt.show()
