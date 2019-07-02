# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:57:24 2019

@author: user
"""

import scipy.signal
import pandas as pd
import scipy.fftpack
import numpy as np
import numpy.fft as fft
from matplotlib.pyplot import *
import plotly.plotly as py
import math
#from hmmlearn import hmm
data=pd.read_csv('t1.csv')

data.head(1)

c=54            #No of gait cycles
l1,l2,l3,l4,l5=[],[],[],[],[]
for i in range(data.shape[0]-1):
    if(data['event'][i]=='HO'):
        l1.append(data['frame_no'][i+1]-data['frame_no'][i])
    elif(data['event'][i]=='TO'):
        l2.append(data['frame_no'][i+1]-data['frame_no'][i])
    elif(data['event'][i]=='MS'):
        l3.append(data['frame_no'][i+1]-data['frame_no'][i])
    elif(data['event'][i]=='HS'):
        l4.append(data['frame_no'][i+1]-data['frame_no'][i])
    elif(data['event'][i]=='FF'):
        l5.append(data['frame_no'][i+1]-data['frame_no'][i])

n1,n2,n3,n4,n5=sum(l1),sum(l2),sum(l3),sum(l4),sum(l5)
n=[n1,n2,n3,n4,n5]
a12,a23,a34,a45,a51=c/n1,c/n2,c/n3,c/n4,c/n5

a11,a22,a33,a44,a55=1-a12,1-a23,1-a34,1-a45,1-a51

tm=[[a11,a12,0,0,0],[0,a22,a23,0,0],[0,0,a33,a34,0],[0,0,0,a44,a45],[a51,0,0,0,a55]]

ntotal=1916

p1,p2,p3,p4,p5=n1/ntotal,n2/ntotal,n3/ntotal,n4/ntotal,n5/ntotal
p=sum([p1,p2,p3,p4,p5])
#emmision probability
zsum=0
for i in range(data.shape[0]-1):
    zsum=zsum+i

sig=[]
mu=[zsum/n1,zsum/n2,zsum/n3,zsum/n4,zsum/n5]
c=0
'''for i in range(5):
    for j in range(data.frame_no.shape[0]-1):
        c=c+(data['frame_no'][j]-mu[i])*(data['frame_no'][j]-mu[i])
    sig.append(math.sqrt((1/(n[i]-1))*c))
    c=0'''
for i in range(5):
    for j in range(data.frame_no.shape[0]-1):
        c=c+((i-mu[i])*(i-mu[i]))
    sig.append(math.sqrt((1/(n[i]-1))*c))
    c=0
    
from hmmlearn import hmm
import numpy as np

model = hmm.MultinomialHMM(n_components=5)
model.startprob_ = np.array([p1,p2,p3,p4,p5])
model.transmat_ = np.array([tm[0],tm[1],tm[2],tm[3],tm[4]])
model.emissionprob_ = np.array([[0.1, 0.4, 0.5],
                                [0.6, 0.3, 0.1],[0.4, 0.5, 0.1],[0.6, 0.2, 0.2],[0.6, 0.3, 0.1]])   

logprob, seq = model.decode(np.array([[1,2,0]]).transpose())
print(math.exp(logprob))
print(seq)


'''import numpy as np
from hmmlearn import hmm

X1 = np.random.random((10, 2))
X2 = np.random.random((5, 2))

X = np.concatenate([X1, X2])
lengthsX = np.array([len(X1), len(X2)])

modelX = hmm.GaussianHMM(n_components=3).fit(X, lengths=lengthsX)

print(X.shape)
# (15L, 2L)

print(modelX.decode(X, algorithm="viterbi"))
# (7.622577397336711, array([1, 0, 1, 0, 1, 2, 0, 0, 0, 0, 1, 1, 2, 2, 2]))'''
        