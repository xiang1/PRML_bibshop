# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:37:21 2018

@author: Liu-pc
"""

import numpy as np
import math
import matplotlib.pyplot as plt
size_s, size_n = 100,50;
data_x_mat = np.linspace(0,1,50);
data_x_mat_noNoise = np.sin(data_x_mat*2*math.pi);
data_y_mat = np.sin(data_x_mat*2*math.pi) + 0.5*np.random.randn(size_s,size_n);

#plt.plot(data_x_mat,data_y_mat[0,:] )

def Cal_Phi_x(x,M):
    Phi_x = np.ones(np.size(x));
    sigma = 0.05;
    mu = np.linspace(0,1,M);
    for i in mu:
        Phi_x = np.c_[Phi_x, np.exp(-(x-i)**2/2/sigma**2)]
    return Phi_x
M = 25;
Phi = Cal_Phi_x(data_x_mat,M);
# 第一种情况
lambda_reg = np.exp(2.6)*np.eye(M+1, dtype=int)  ;
W = np.dot(np.dot(np.linalg.inv((lambda_reg+np.dot(Phi.transpose(), Phi))),Phi.transpose()),data_y_mat.transpose());
y_predict = np.dot(Phi, W);
#plt.plot(data_x_mat,y_predict[:,1:20]);
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.plot(data_x_mat,y_predict[:,1:20],label='Predict');
ax2.plot(data_x_mat,np.mean(y_predict,1),data_x_mat,data_x_mat_noNoise)
ax2.legend(['Predict','Real Model'],loc='right');
ax1.text(0.7,0.8, r'$\lambda = \exp(2.6)$')
ax1.set_xlabel(r'$x$', fontsize="xx-large")
ax2.set_xlabel(r'$x$', fontsize="xx-large")
plt.show();

# 第二种情况
lambda_reg = np.exp(-0.31)*np.eye(M+1, dtype=int)  ;
W = np.dot(np.dot(np.linalg.inv((lambda_reg+np.dot(Phi.transpose(), Phi))),Phi.transpose()),data_y_mat.transpose());
y_predict = np.dot(Phi, W);
#plt.plot(data_x_mat,y_predict[:,1:20]);
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.plot(data_x_mat,y_predict[:,1:20],label='Predict');
ax2.plot(data_x_mat,np.mean(y_predict,1),data_x_mat,data_x_mat_noNoise)
ax2.legend(['Predict','Real Model'],loc='right');
ax1.text(0.6,0.8, r'$\lambda = \exp(-0.31)$')
ax1.set_xlabel(r'$x$', fontsize="xx-large")
ax2.set_xlabel(r'$x$', fontsize="xx-large")
plt.show();
# 第三种情况
lambda_reg = np.exp(-2.4)*np.eye(M+1, dtype=int)  ;
W = np.dot(np.dot(np.linalg.inv((lambda_reg+np.dot(Phi.transpose(), Phi))),Phi.transpose()),data_y_mat.transpose());
y_predict = np.dot(Phi, W);
#plt.plot(data_x_mat,y_predict[:,1:20]);
fig, (ax1, ax2) = plt.subplots(figsize=(8, 3), ncols=2)
ax1.plot(data_x_mat,y_predict[:,1:20],label='Predict');
ax2.plot(data_x_mat,np.mean(y_predict,1),data_x_mat,data_x_mat_noNoise)
ax2.legend(['Predict','Real Model'],loc='right');
ax1.text(0.6,0.8, r'$\lambda = \exp(-2.4)$')
ax1.set_xlabel(r'$x$', fontsize="xx-large")
ax2.set_xlabel(r'$x$', fontsize="xx-large")
plt.show();





