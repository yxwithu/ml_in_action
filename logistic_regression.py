
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

def sigmoid(x):
    return 1.0 / (1 +  np.exp(-x))


# In[ ]:

def batch_grad_ascent(X_train, y_train, alpha, n_iter):
    X_train = np.mat(X_train)
    y_train = np.mat(y_train).transpose()
    m, n = shpae(X_train)
    weighs = np.ones((n, 1))
    for k in range(n_iter):
        hx = sigmoid(weighs * X_train)
        error = y_train - hx
        weighs += alpha * X_train.transpose() * error
    return weights


# In[ ]:

def stoc_grad_ascent(X_train, y_train, alpha):
    m, n = shape(X_train)
    weights = np.ones((n, 1))
    for i in range(m):
        h = sigmoid(np.sum(weights * X_train[i]))
        error = y_train[i] - h
        weights += alpha * error * X_train[i]
    return weights


# In[ ]:



