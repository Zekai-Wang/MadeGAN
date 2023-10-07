import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import interpolate
from biosppy.signals import ecg
import wfdb
import os
import time

def bisearch(key, array):
    '''
    search value which is most closed to key
    :param key:
    :param array:
    :return:
    '''
    lo = 0
    hi = len(array)-1
    while lo <= hi:
        mid = lo + int((hi - lo) / 2)
        if key <array[mid]:
            hi = mid - 1
        elif key>array[mid] :
            lo = mid + 1
        else:
            return array[mid]
    if hi<0:
        return array[0]
    if lo>=len(array):
        return array[-1]
    return array[hi] if (key-array[hi])<(array[lo]-key) else array[lo]


# In[3]:


PATIENTS = ['s20011', 's20021','s20031', 's20041', 's20051', 's20061',
            's20071', 's20081','s20091', 's20101', 's20111', 's20121',
            's20131', 's20141','s20151', 's20161', 's20171', 's20181',
            's20191', 's20201','s20211', 's20221', 's20231', 's20241']
DATA_DIR = "/Users/zewang/Desktop/long-term-st-database-1.0.0/"


N_samples=[]
V_samples=[]
S_samples=[]
F_samples=[]
Q_samples=[]

N={"N","L","R"}
S={"a", "J", "A", "S", "j", "e"}
V={"V","E"}
F={"F"}
Q={"/", "f", "Q"}
BEAT=N.union(S,V,F,Q)
ABNORMAL_BEAT=S.union(V,F,Q)

fs = 250
fs_rs = 308

LEFT=120
RIGHT=140

def preprocess(idx):

    x1,x2 = wfdb.rdsamp(os.path.join(DATA_DIR, str(idx)))
    annotation = wfdb.rdann(os.path.join(DATA_DIR, str(idx)), 'atr',summarize_labels=True)
    sig_out=ecg.ecg(signal=x1[:,0], sampling_rate=250., show=False)
    sig=sig_out["filtered"]
    r_peaks=sig_out["rpeaks"]
    ann_types = annotation.symbol
    ann_signal_idx = annotation.sample
    
    return sig, r_peaks, ann_types, ann_signal_idx


# In[106]:


for idx in PATIENTS:
    start_time = time.time()
    
    sig, r_peaks, ann_types, ann_signal_idx = preprocess(idx)
    samples=[]

    for ann_idx, ann_type in enumerate(ann_types):
        if ann_type in BEAT:
            sig_idx=ann_signal_idx[ann_idx]
            if sig_idx-LEFT>=0 and sig_idx+RIGHT<len(sig):
                if ann_type in N:
                    if ann_idx <= 10000:
                        closed_rpeak_idx=bisearch(sig_idx,r_peaks)
                        if abs(closed_rpeak_idx-sig_idx)<10:
                            samples.append((sig[sig_idx-LEFT:sig_idx+RIGHT],'N',ann_type))
                else:
                    AAMI_label=""
                    if ann_type in S:
                        AAMI_label = "S"
                    elif ann_type in V:
                        AAMI_label = "V"
                    elif ann_type in F:
                        AAMI_label = "F"
                    elif ann_type in Q:
                        AAMI_label="Q"
                    else:
                        raise  Exception("annonation type error")
                    assert AAMI_label!=""
                    samples.append((sig[sig_idx - LEFT:sig_idx + RIGHT], AAMI_label, ann_type))
    # Get time array
    times = np.arange(len(samples[0][0])) * 1 / fs
    
    # Generate new resampling time array
    times_rs = np.arange(0, times[-1], 1 / fs_rs)
    
    for sample in samples:
        interp_func = interpolate.interp1d(x=times, y=sample[0], kind='linear')
        values_rs = interp_func(times_rs)
        if sample[1]=="N":
            N_samples.append(values_rs)
        elif sample[1]=="S":
            S_samples.append(values_rs)
        elif sample[1]=="V":
            V_samples.append(values_rs)
        elif sample[1]=="F":
            F_samples.append(values_rs)
        elif sample[1]=="Q":
            Q_samples.append(values_rs)
        else:
            raise  Exception("sample AAMI type error, input type: {}".format(sample[1]))
            
    cur_time = time.time()
    print('Finish patient', idx, 'pre-processing', '|| Time:', int((cur_time - start_time)/60), 'min')
    print('The Number of beats ||',"N:", len(N_samples), "V:", len(V_samples), "S:", len(S_samples),"F:", len(F_samples), "Q:", len(Q_samples))
    print('=============================='*2)


N_samples_LTST=np.array(N_samples)
V_samples_LTST=np.array(V_samples)
S_samples_LTST=np.array(S_samples)
F_samples_LTST=np.array(F_samples)


np.random.shuffle(N_samples_LTST)
np.random.shuffle(V_samples_LTST)
np.random.shuffle(S_samples_LTST)
np.random.shuffle(F_samples_LTST)


np.save('N_samples_LTST.npy',N_samples_LTST)
np.save('V_samples_LTST.npy',V_samples_LTST)
np.save('S_samples_LTST.npy',S_samples_LTST)
np.save('F_samples_LTST.npy',F_samples_LTST)

