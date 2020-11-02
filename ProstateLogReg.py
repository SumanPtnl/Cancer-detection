#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
data = pd.read_csv(r'Prostate_Cancer.csv')
data.drop(["id"],axis=1,inplace = True)
# we convert data type the float.
c = ['radius','texture','perimeter','area']

for col in c:
    data[col] = data[col].astype(float)
    #Convert two class names to 0 and 1
data.diagnosis_result = [1 if each == "M" else 0 for each in data.diagnosis_result]
#assign Class_att column as y attribute
y = data.diagnosis_result.values

#drop Class_att column, remain only numerical c
x_data = data.drop(["diagnosis_result"],axis=1)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
from sklearn.model_selection import train_test_split
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2,random_state =42)
#transpose matrices
xtr = xtr.T
ytr = ytr.T
xte = xte.T
yte = yte.T
#parameter initialize and sigmoid function
def init_wt_bias(dimension):
   
    w = np.full((dimension,1),0.01) #first initialize w values to 0.01
    b = 0.0 #first initialize bias value to 0.0
    return w,b
#sigmoid function fits the z value between 0 and 1
def sigmoid(z):
   
    y_head = 1/(1+ np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,xtr,ytr):
    # forward propagation
   
    y_head = sigmoid(np.dot(w.T,xtr) + b)
    loss = -(ytr*np.log(y_head) + (1-ytr)*np.log(1-y_head))
    cost = np.sum(loss) / xtr.shape[1]
   
    # backward propagation
    derivative_weight = np.dot(xtr,((y_head-ytr).T))/xtr.shape[1]
    derivative_bias = np.sum(y_head-ytr)/xtr.shape[1]
    grad = {"derivative_weight" : derivative_weight, "derivative_bias" : derivative_bias}
   
    return cost,grad
def update_weight_and_bias(w,b,xtr,ytr,learning_rate,iter_num) :
    cost_list = []
    index = []
   
    #for each iteration, update w and b values
    for i in range(iter_num):
        cost,grad = forward_backward_propagation(w,b,xtr,ytr)
        w = w - learning_rate*grad["derivative_weight"]
        b = b - learning_rate*grad["derivative_bias"]
       
        cost_list.append(cost)
        index.append(i)

    para = {"weight": w,"bias": b}
   
    print("iter_num:",iter_num)
    print("cost:",cost)

    #plot cost versus iteration graph to see how the cost changes over number of iterations
    plt.plot(index,cost_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()

    return para, grad
def pred(w,b,xte):
    z = np.dot(w.T,xte) + b
    y_pred_head = sigmoid(z)
   
    #create new array with the same size of x_test and fill with 0's.
    y_pred = np.zeros((1,xte.shape[1]))
   
    for i in range(y_pred_head.shape[1]):
        if y_pred_head[0,i] <= 0.3:
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 1
    return y_pred
def log_reg(xtr,ytr,xte,yte,learning_rate,iter_num):
    dimension = xtr.shape[0]#For our dataset, dimension is 248
    w,b = init_wt_bias(dimension)
   
    para, grad = update_weight_and_bias(w,b,xtr,ytr,learning_rate,iter_num)

    y_pred = pred(para["weight"],para["bias"],xte)
   
    # Print test Accuracy
   
    print("manual test accuracy:",(100 - np.mean(np.abs(y_pred - yte))*100)/100)
log_reg(xtr,ytr,xte,yte,4,400)


# In[ ]:




