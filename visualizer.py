import pickle
import numpy as np
import matplotlib.pyplot as plt
from eeg_generator import EEG_Generator_Izhikevitch
import scipy.signal as signal

def load_pickle(path):
    return pickle.load(open(path, "rb"))


def plot_UAF_list(file_path, label, color='r', opacity=0.8):
    UAF_list = pickle.load(open(file_path, "rb"))
    T = np.min(UAF_list)
    res = 0
    while res<0.5:
        res = (np.array(UAF_list)<T).sum()/len(UAF_list)
        T+=1
    print(T)
    UAF_list_avg = np.mean(UAF_list)
    plt.hist(UAF_list, bins=20, label = label, alpha=opacity, color=color, density = True)
    plt.axvline(UAF_list_avg, c=color)

plot_UAF_list('saved_data/UAF_list_baseline_5mn.npy', 'Inactive target')
plot_UAF_list('saved_data/UAF_list_baseline+1_5mn.npy', 'Active target', color='b')
plt.axvline(137, c='g', label='Optimal threshold')
plt.xlabel('UAF value')
plt.ylabel('Frequency')
plt.legend()
plt.show()

final60= np.minimum(1, np.array(load_pickle('saved_data/final_cont_60.npy')))
finalbin= np.minimum(1, np.array(load_pickle('saved_data/final_bin.npy')))
final20= np.minimum(1, np.array(load_pickle('saved_data/final_cont_20.npy')))
UAF_list_tar0 = pickle.load(open("saved_data/UAF_list_baseline_5mn.npy", "rb"))
UAF_list_tar1= pickle.load(open("saved_data/UAF_list_baseline+1_5mn.npy", "rb"))

min_T = np.min(UAF_list_tar0)
max_T = np.max(UAF_list_tar1)
X = np.linspace(min_T, max_T, 30)
plt.plot(X,finalbin, label='Binary')
plt.plot(X,final20, label='$\\tau$=20')
plt.plot(X,final60, label='$\\tau$=60')
plt.legend()
plt.xlabel('UAF threshold')
plt.ylabel('Final target weight')
plt.show()
pos_nf = pickle.load(open("saved_data/pos_list_after_nf.npy", "rb"))
rank = 1000-np.array(pos_nf)
mean_size = 5000
rank_avg = np.convolve(rank, np.ones(mean_size)/mean_size, mode='valid')
plt.plot(rank_avg)
plt.xlabel('Iterations')
plt.ylabel('MSN unit activation rate rank')
plt.show()

eeg_baseline = pickle.load(open("saved_data/eeg_gen_baseline_5mn.npy", "rb"))
eeg_nf =pickle.load(open("saved_data/eeg_gen_after_nf.npy", "rb"))
baseline_eeg = eeg_baseline.eeg_history
# plt.plot(baseline_eeg[-1000:])
# plt.xlabel('t (ms)')
# plt.ylabel('EEG signal')
# plt.show()
#
nf_eeg = eeg_nf.eeg_history
# plt.plot(nf_eeg[-1000:])
# plt.xlabel('t (ms)')
# plt.ylabel('EEG signal')
# plt.show()
def display_PAF(eeg_time, frame = [None,None], label=None, alpha = 1):
    fs, Pxx = signal.periodogram(eeg_time[frame[0]:frame[1]], fs=10**3, window=np.hamming(1024))
    cubic_Pxx = np.cbrt(Pxx)
    plt.plot(fs[:40], cubic_Pxx[:40], label=label, alpha=alpha)
start = -4000
range = [start,start+1024]
display_PAF(baseline_eeg, range , 'Baseline', alpha = 0.3)
display_PAF(nf_eeg, range, 'Neurofeedback')
plt.xlabel('EEG Frequency')
plt.ylabel('Power')
plt.legend()
plt.show()
