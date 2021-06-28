import numpy as np
import pickle
import matplotlib.pyplot as plt

eta = 0.0001
UAF_tar0 = np.array(pickle.load(open("saved_data/UAF_list_baseline_5mn.npy", "rb")))
UAF_tar1= np.array(pickle.load(open("saved_data/UAF_list_baseline+1_5mn.npy", "rb")))

def get_CDF(dist, T):
    return (dist < T).sum()/len(dist)

L = []
valid_T = []
for T in np.linspace(UAF_tar0.min(), UAF_tar0.max(), 100):
    try:
        F0 = get_CDF(UAF_tar0, T)
        F1 = get_CDF(UAF_tar1, T)
        Ps = [0.001]
        for t in range(10000):
            p = Ps[-1]
            p_next = p*((1-F1)*(p+eta)/(1+10*eta) +F1*(p-eta)/(1-10*eta)) + (1-p) * (F0*p/(1-10*eta) + (1-F0)*p/(1+10*eta))
            Ps.append(p_next)
        plt.plot(Ps)
        b = 2*(F1+F0) + 9*eta
        c = 2*F0 + 10*eta - 1
        assert((b**2-4*c)>0)
        l = (-b+np.sqrt(b**2-4*c))/2
        valid_T.append(T)
        L.append(l)
    except Exception:
        pass
plt.show()
plt.plot(valid_T, L)
plt.show()
