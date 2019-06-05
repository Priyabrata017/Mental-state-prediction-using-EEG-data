
"""
Created on Mon May 27 18:52:37 2019

@author: Priyabrata
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import cmath

data=pd.read_csv("eeg_data.csv")
#data=dat.interpolate(method='linear', limit_direction='forward', axis=0)
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

from scipy.signal import freqz

fs = 128
lowcut = 3
highcut = 40
y=[]
z=[]
a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14=[],[],[],[],[],[],[],[],[],[],[],[],[],[]
for i in range(5761):
    a1.append(data.f1[i])
    a2.append(data.f2[i])
    a3.append(data.f3[i])
    a4.append(data.f4[i])
    a5.append(data.f5[i])
    a6.append(data.f6[i])
    a7.append(data.f7[i])
    a8.append(data.f8[i])
    a9.append(data.f9[i])
    a10.append(data.f10[i])
    a11.append(data.f11[i])
    a12.append(data.f12[i])
    a13.append(data.f13[i])
    a14.append(data.f14[i])
    
from scipy.fftpack import fft, ifft
a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14 = np.array(a1),np.array(a2),np.array(a3),np.array(a4),np.array(a5),np.array(a6),np.array(a7),np.array(a8),np.array(a9),np.array(a10),np.array(a11),np.array(a12),np.array(a13),np.array(a14)
b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14 = fft(a1),fft(a2),fft(a3),fft(a4),fft(a5),fft(a6),fft(a7),fft(a8),fft(a9),fft(a10),fft(a11),fft(a12),fft(a13),fft(a14) 
c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14=b1.real,b2.real,b3.real,b4.real,b5.real,b6.real,b7.real,b8.real,b9.real,b10.real,b11.real,b12.real,b13.real,b14.real
lists=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14]

fft_vals=[None]*14
fft_freq=[None]*14
freq_ix=[[0]*2881]*14
fs = 128                               # Sampling rate (128 Hz)
for i in range(14):
    fft_vals[i]=(np.absolute(np.fft.rfft(lists[i])))
    # Get real amplitudes of FFT (only in postive frequencies)
for i in range(14):
        fft_freq[i]= np.fft.rfftfreq(len(lists[i]), 1.0/fs)

# Define EEG bands
eeg_bands = {'Alpha': (7, 14),
             'Beta': (14, 30)}

eeg_band_fft = dict()
for i in range(14):
    for j in range(2881):
        for band in eeg_bands:
            freq_ix[i][j] = np.where((fft_freq[i][j] >= eeg_bands[band][0]) & (fft_freq[i][j] <= eeg_bands[band][1]))[0]
            
        #eeg_band_fft[band] = np.mean(fft_vals[freq_ix[i]])
p=[]                       
for i in range(14):
    p.append(float(np.mean(fft_vals[i])))
for i in range(14):
    for j in range(len(fft_vals[i])):
        fft_vals[i][j]=fft_vals[i][j]/p[i]
        
import pandas as pd
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")

df=[None]*14
for i in range(14):
    df[i]=pd.DataFrame(fft_vals[i],columns=['f'])




dat=b.interpolate(method='linear', limit_direction='forward', axis=0)
'''df=pd.DataFrame(y,columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14'])   
df.to_csv("fffilter.csv",index=False)  '''