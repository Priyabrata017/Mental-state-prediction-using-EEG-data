# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 19:53:55 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error as rms
from collections import Counter

import statistics
def find_max_mode(list1):
    list_table = statistics._counts(list1)
    len_table = len(list_table)

    if len_table == 1:
        max_mode = statistics.mode(list1)
    else:
        new_list = []
        for i in range(len_table):
            new_list.append(list_table[i][0])
        max_mode = max(new_list) # use the max value here
    return max_mode

# Importing the dataset
data = pd.read_csv('class2.csv')

X = data.iloc[:, 0:14].values
y = data.iloc[:, 14].values


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier1.fit(X_train, y_train)

from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2.fit(X_train, y_train)

from sklearn.svm import SVC
classifier3 = SVC(kernel = 'rbf', random_state = 0)
classifier3.fit(X_train, y_train)

y_pred1= classifier1.predict(X_test)
y_pred2= classifier2.predict(X_test)
y_pred3= classifier3.predict(X_test)

y_pred = []
a0,a1,a2=0,0,0
for i in range(0,len(X_test)):
    p=[y_pred1[i],y_pred2[i],y_pred3[i]]
    for j in p:
        if(j==0):
            a0=a0+1
        elif(j==1):
            a1=a1+1
        else:
            a2=a2+1
    if(a0>a1 and a0>a2):
        y_pred.append(0)
    elif(a1>a0 and a1>a2):
        y_pred.append(1)
    else:
        y_pred.append(2)
    a0,a1,a2=0,0,0

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)       
        
            
   
   
