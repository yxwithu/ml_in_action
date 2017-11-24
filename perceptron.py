
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

def fit(X, y, eta, n_iter):
    """standard form """
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    
    sample_list = np.arange(m)
    
    i = 0
    while i < n_iter:
        i+= 1
        
        wrong_flag = False
        
        np.random.shuffle(sample_list)  #打乱顺序
    
        for sample_id in sample_list:
            feat_vec = X[sample_id]
            label = y[sample_id]

            if (np.sum(weights * feat_vec) + bias) * label <= 0:  #分类错误点
                weights += eta * label * feat_vec
                bias += eta * label
                
                wrong_flag = True
           
        if not wrong_flag:
            break
        
    print("iter: %d" % i)
    return weights, bias


# In[ ]:

def fit_dual(X, y, eta, n_iter):
    """dual form """
    m, n = X.shape
    alpha = np.zeros(m)
    bias = 0
    sample_list = np.arange(m)
    
    gram_matrix = np.dot(X, X.transpose())  # gram矩阵，用于方便运算
    
    i = 1
    while i <= n_iter:
        i += 1
        wrong_flag = False
        
        np.random.shuffle(sample_list)  #打乱顺序
        
        for sample_id in sample_list:
            feat_vec = X[sample_id]
            label = y[sample_id]
            
            if (np.sum(alpha * y * gram_matrix[sample_id]) + bias) * label <= 0:
                wrong_flag = True
                alpha[sample_id] += eta
                bias += eta * label
            
        if not wrong_flag:
            break
    print("iter : %d" % i)
    return np.dot((alpha * y).transpose(), X), bias


# In[ ]:

X_train = np.array([[3,3],
                    [4,3],
                    [1,1]])
y_train = [1,1,-1]


# In[ ]:

fit_dual(X_train, y_train, 1, 100)

