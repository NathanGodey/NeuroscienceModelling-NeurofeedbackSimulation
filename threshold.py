import numpy as np
import scipy.stats as stats
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
def avg(arr):
    return arr
    return np.cumsum(arr)/np.arange(1,len(arr)+1)

UAF_list_tar0 = pickle.load(open("saved_data/UAF_list_baseline_5mn.npy", "rb"))
UAF_list_tar1= pickle.load(open("saved_data/UAF_list_baseline+1_5mn.npy", "rb"))

np.random.shuffle(UAF_list_tar0)
np.random.shuffle(UAF_list_tar1)

min_T = np.min(UAF_list_tar0)
max_T = np.max(UAF_list_tar0)

target_id = 3
K = 50
n_thresh = 5
sample_rate = 10
learning_rate = 1e-5
target_weights = [[] for i in range(n_thresh)]
timed_zones = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

for i_thresh, threshold in tqdm(enumerate(np.linspace(min_T, max_T, n_thresh))):
    for k in range(K):
        start = time.time()
        msn_weights = 0.001 * np.ones(999)
        weight = 0.001
        i_i_s = 0
        S = msn_weights.sum()
        i_dist_0 = 0
        i_dist_1 = 0
        I_s = np.random.randint(0,989, size=1000)
        for t in range(5000):
            i_s = I_s[i_i_s % len(I_s)]
            i_i_s += 1
            p_target_in_active = 1-(1-weight)**10
            #print(p_target_in_active, weight, S)
            target_in_active = np.random.random() < p_target_in_active
            timed_zones['A'] += time.time() - start
            start = time.time()
            if target_in_active:
                UAF = UAF_list_tar1[i_dist_1 % len(UAF_list_tar1)]
                i_dist_1 += 1
            else:
                UAF = UAF_list_tar0[i_dist_0 % len(UAF_list_tar0)]
                i_dist_0 += 1
            timed_zones['B'] += time.time() - start
            start = time.time()
            len_sel = 10-target_in_active
            if UAF > threshold:
                msn_weights[i_s: i_s+len_sel] += learning_rate
                S += len_sel * learning_rate
                weight += learning_rate
            else:
                modif_mask = (msn_weights[i_s: i_s+len_sel]>learning_rate)
                msn_weights[i_s: i_s+len_sel] -= learning_rate * modif_mask
                S -= learning_rate * modif_mask.sum()
                weight -= learning_rate
            timed_zones['C'] += time.time() - start
            start = time.time()
            #print(msn_weights.sum())
            weight = max(0, weight)
            weight = weight / (S + weight)
            if t % sample_rate == 0:
                if k == 0:
                    target_weights[i_thresh].append(weight/K)
                else:
                    target_weights[i_thresh][t//sample_rate]+=weight/K

            timed_zones['D'] += time.time() - start
            start = time.time()
    plt.plot(avg(target_weights[i_thresh]), color=plt.get_cmap('plasma')(i_thresh/n_thresh))
    timed_zones['E'] += time.time() - start
plt.show()
print(timed_zones)
