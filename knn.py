
# coding: utf-8

# In[2]:

import numpy as np


# In[42]:

def arrayKNN(inX, X_train, y_train, k):
    n_samples = X_train.shape[0]
    
    dist_mat = (np.tile(inX, (n_samples, 1)) - X_train) ** 2  #距离矩阵
    dist_sum = np.sum(dist_mat, axis = 1)  #距离之和
    sorted_dist_idx = dist_sum.argsort()   #按数值排序后的原index所在位置
    class_cnt_dict = {}
    for i in range(k):
        vote_class = y_train[sorted_dist_idx[i]]
        class_cnt_dict[vote_class] = class_cnt_dict.get(vote_class, 0) + 1
    sorted_class_cnt =  [(k, class_cnt_dict[k]) for k in sorted(class_cnt_dict, key=class_cnt_dict.get, reverse=True)]  #取其中一个
    return sorted_class_cnt[0][0]


# In[23]:

def createDataSet():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels