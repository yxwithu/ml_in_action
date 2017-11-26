
# coding: utf-8

# In[1]:

import numpy as np


# In[3]:

def trainNB(X_train, y_train):
    """二分类朴素贝叶斯训练，得到概率值"""
    doc_cnt = len(y_train)
    word_cnt = len(X_train[0])
    
    pAbusive = np.sum(y_train) / doc_cnt  
    word_p0_cnt, word_p1_cnt = np.ones(word_cnt), np.ones(word_cnt)  #拉普拉斯平滑
    word_p0_sum, word_p1_sum = 2, 2  #拉普拉斯平滑
    for i in range(doc_cnt):
        if y_train[i] == 1:
            word_p1_cnt += X_train[i]
            word_p1_sum += np.sum(X_train[i])
        else:
            word_p0_cnt += X_train[i]
            word_p0_sum += np.sum(X_train[i])
    p1_vec = word_p1_cnt / word_p1_sum
    p0_vec = word_p0_cnt / word_p0_sum
    return np.log(p1_vec), np.log(p0_vec), pAbusive   #防止下溢


# In[4]:

def classifyNB(X_test, p0_vec, p1_vec, pAbusive):
    p1 = np.sum(X_test * p1_vec) + np.log(pAbusive)  #防止下溢
    p0 = np.sum(X_test * p0_vec) + np.log(1 - pAbusive)
    return 1 if p1 > p0 else 0

