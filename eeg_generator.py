import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.fft as fft
import scipy.signal as signal
from neurolib.models.aln import ALNModel
import neurolib.utils.functions as func
import pickle

def extract_UAF(eeg_time, frame = [None,None]):
    fs, Pxx = signal.periodogram(eeg_time[frame[0]:frame[1]], fs=10**3, window=np.hamming(1024))
    cubic_Pxx = np.cbrt(Pxx)
    # plt.plot(fs[:40], cubic_Pxx[:40])
    # plt.show()
    PAF = 8+np.argmax(cubic_Pxx[8:13])
    UAF = np.mean(cubic_Pxx[PAF:PAF+3])
    return UAF, PAF

class EEG_Generator_Izhikevitch:
    def __init__(self, nb_exc, nb_inh):
        self.nb_total = nb_exc + nb_inh
        self.nb_exc = nb_exc
        self.nb_inh = nb_inh
        self.re = np.random.random(nb_exc)
        self.ri = np.random.random(nb_inh)

        self.a = 0.02 * np.ones(self.nb_total)
        self.a[nb_exc:] += 0.08*self.ri

        self.b = 0.2 * np.ones(self.nb_total)
        self.b[nb_exc:] += 0.05*(1-self.ri)

        self.c = -65 * np.ones(self.nb_total)
        self.c[:nb_exc] += 15*self.re**2

        self.d = 2 * np.ones(self.nb_total)
        self.d[:nb_exc] += 6*(1-self.re**2)

        self.S = np.random.random((self.nb_total, self.nb_total))
        self.S[:, :nb_exc] *= 0.5
        self.S[:, nb_exc:] *= -1

        self.v = -65*np.ones(self.nb_total)
        self.u = self.b*self.v
        self.eeg_history = [0]
        self.v_history = []
        self.max_memory = 6000

    def step(self, thalamic_inp_add=0):
        I = np.random.randn(self.nb_total)
        I[:self.nb_exc] *= 5
        I[self.nb_exc:] *= 2
        I[:self.nb_exc] += thalamic_inp_add
        fired = np.argwhere(self.v>=30).flatten()
        self.v[fired] = self.c[fired]
        self.u[fired] += self.d[fired]
        if len(fired)>0:
            I += np.sum(self.S[:,fired],axis = 1).flatten()
        self.v += 0.5*(0.04*self.v**2 + 5*self.v + 140 - self.u + I)
        self.v += 0.5*(0.04*self.v**2 + 5*self.v + 140 - self.u + I)
        self.u += self.a * (self.b*self.v - self.u)

        self.v_history.append([i for i in self.v])
        self.eeg_history += [0.9*self.eeg_history[-1] + 0.1*self.v[:self.nb_exc].sum()]

        if len(self.v_history) > self.max_memory:
            self.v_history.pop(0)
            self.eeg_history.pop(0)


if __name__=='__main__':
    gen = EEG_Generator(800,100)
    UAFs = []
    for t in tqdm(range(1,300000)):
        gen.step(1)
        if t%1024==0:
            UAFs.append(extract_UAF(gen.eeg_history, [-1024, None])[0])
    pickle.dump(UAFs, open('saved_data/UAF_list_800v100+1.npy', 'wb'))

    # plt.plot(gen.eeg_history[300:1324])
    # plt.show()
    # fft_eeg = np.abs(fft.fft(np.hamming(1024)*gen.eeg_history[300:1324]))
    # plt.plot(fft_eeg[2:50])
    # plt.show()
    # v_hist = np.array(gen.v_history)
    # print(gen.a[5], gen.b[5], gen.c[5], gen.d[5])
    # plt.plot(v_hist[:1000, 5])
    # plt.show()
    # print(gen.a[805], gen.b[805], gen.c[805], gen.d[805])
    # plt.plot(v_hist[:1000,805])
    # plt.show()
