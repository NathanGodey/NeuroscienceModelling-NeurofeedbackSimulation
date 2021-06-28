import scipy.fft as fft
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

from eeg_generator import EEG_Generator_Izhikevitch
from msn import MSN_Units

import os, psutil
process = psutil.Process(os.getpid())

eeg_gen = EEG_Generator_Izhikevitch(800,200)
msn_units = MSN_Units(1000, 10/1000)
target_unit = 3
chunksize = 1024
delta_update = 100
TBR_thresh = 60
duration = 10*60*1000
save_suffix = 'after_nf_tbr'
feedback_mode = True
TBR_list = []
list_pos = []
chunk_prog_bar = tqdm(range(0, duration))

def get_ranks(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def extract_TBR(eeg_time, frame = [None,None]):
    fs, Pxx = signal.periodogram(eeg_time[frame[0]:frame[1]], fs=10**3, window=np.hamming(1024))
    cubic_Pxx = np.cbrt(Pxx)
    # plt.plot(fs[:40], cubic_Pxx[:40])
    # plt.show()
    theta = np.mean(cubic_Pxx[4:7])
    beta = np.mean(cubic_Pxx[13:30])
    TBR = theta/beta
    return TBR, (theta, beta)

for t in chunk_prog_bar:
    if t == duration //2:
        feedback_mode = False
    #Initialization
    if t<chunksize:
        chunk_prog_bar.set_description(f't: {t//60000}:{(t%60000)//1000} (init)')
        eeg_gen.step(msn_units.state[target_unit])
        if feedback_mode:
            msn_units.update_weight()
            S = msn_units.weights.sum()
            msn_units.normalize(S)
            msn_units.update_state()
    else:

        # Initial feedback
        t_chunk = t%chunksize
        if t == chunksize:
            current_UAF, current_PAF = extract_UAF(eeg_gen.eeg_history, [-1024,None])
            feedback = 1-2*int(current_UAF < UAF_thresh)
        pos_target = get_ranks(np.mean(msn_units.history, axis=1))[target_unit]
        list_pos.append(pos_target)
        chunk_prog_bar.set_description( \
        f't: {t//60000}:{(t%60000)//1000}; UAF: {current_UAF}, PAF: {current_PAF}; pos: {pos_target} (ram: {process.memory_info().rss//10**6}Mo)')

        eeg_gen.step(msn_units.state[target_unit])
        S = 0
        if t > 0 and (t%delta_update)==0:
            current_UAF, current_PAF = extract_UAF(eeg_gen.eeg_history, [-1024,None])
            UAF_list.append(current_UAF)
            if feedback_mode:
                feedback = 1-2*int(current_UAF < UAF_thresh)
                msn_units.update_weight(feedback = feedback)
                S = msn_units.weights.sum()
        else:
            if feedback_mode:
                msn_units.update_weight()
                S = msn_units.weights.sum()
        if feedback_mode:
            msn_units.normalize(S)
            msn_units.update_state()

pickle.dump(UAF_list, open(f"saved_data/UAF_list_{save_suffix}.npy", "wb"))
pickle.dump(msn_units, open(f"saved_data/msn_{save_suffix}.npy", "wb"))
pickle.dump(eeg_gen, open(f"saved_data/eeg_gen_{save_suffix}.npy", "wb"))
pickle.dump(list_pos, open(f"saved_data/pos_list_{save_suffix}.npy", "wb"))
mean_UAF, std_UAF = np.mean(UAF_list), np.std(UAF_list)
print(mean_UAF, std_UAF)
UAF_list_no_outlier = [uaf for uaf in UAF_list if abs(uaf-mean_UAF)<2*std_UAF]
plt.hist(UAF_list_no_outlier, bins=50, density = True)
plt.show()
