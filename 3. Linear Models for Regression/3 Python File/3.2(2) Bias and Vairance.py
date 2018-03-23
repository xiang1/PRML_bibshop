# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 22:31:19 2018

@author: Liu-pc
"""

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
lambda_mat = np.linspace(np.exp(-3), np.exp(2),50);
bias = np.zeros((50,1)); 
variance = np.zeros((50,1));
test_error = np.zeros((50,1));
i = 0;
for lambda_a in lambda_mat:
  lambda_reg = lambda_a*np.eye(M+1, dtype=int)  ;
  W = np.dot(np.dot(np.linalg.inv((lambda_reg+np.dot(Phi.transpose(), Phi))),Phi.transpose()),data_y_mat.transpose());
  y_predict = np.dot(Phi, W);  
  bias[i,0] = np.mean((np.mean(y_predict,1)-data_x_mat_noNoise)**2);
  mean_y_pre_mat =  np.tile(np.mean(y_predict,1),  (np.size(y_predict,1),1))
  variance[i,0] = np.sum(np.sum((y_predict-mean_y_pre_mat.transpose())**2))/y_predict.size;
  test_error[i,0] = np.sum(np.sum((y_predict-data_y_mat.transpose())**2))/y_predict.size;
  i=i+1;

lambda_mat = lambda_mat.reshape(50,1)
plt.plot(np.log(lambda_mat), bias,'b-',label='bias');
plt.plot(np.log(lambda_mat), variance,'r-', label = 'variance');
plt.plot(np.log(lambda_mat), test_error,'g-', label='test_error') ;
plt.plot(np.log(lambda_mat), bias+ variance,'c-',label = 'bias + variance');
plt.xlabel('$\ln \lambda$');
plt.legend(loc="center left");
plt.show()

#plt.legend();
  
  
