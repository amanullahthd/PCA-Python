# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:37:07 2021

@author: amanullah.awan
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

def pca_comp(x):
    m = np.mean(x , axis = 0)
    x_centered = x - m
    
    x_cov = np.cov(x_centered.T)
    eigen_vals , eigen_vec = np.linalg.eig(x_cov)
    
    i= np.argsort(eigen_vals)[::-1]
    eigen_vec = eigen_vec[:,i]
    eigen_vals = eigen_vals[i]
    
    return (eigen_vals , eigen_vec , m)

iris = datasets.load_iris()

x = iris.data
y = iris.target
n = 2

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)

X_evals, X_evecs, X_mean = pca_comp(X_train)


X_evecs_n = X_evecs[:,:n]

X_factors_train = np.dot(X_train-X_mean,X_evecs_n)
X_factors_test= np.dot(X_test-X_mean,X_evecs_n)

print("Training Set Dimensions:", X_train.shape)
print("Test Set Dimensions:", X_test.shape)
print("Training Set Dimensions after PCA:", X_factors_train.shape)
print("Test Set Dimensions after PCA:", X_factors_test.shape)


    