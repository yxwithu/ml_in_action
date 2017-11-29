
# coding: utf-8

# 基于单层决策树的adaboost，这个单层决策树跟平时用的单层决策树不同，这里只是考察某个特征的某个值，根据与这个值比较结果的大小不同分类，而非普通决策树中的投票法确定类别。

# In[ ]:

import numpy as np


# In[ ]:

def stump_classify(data_mat, dimen, thresh, thresh_ineq):
    """将数据第dimen维特征与thresh比较，根据大小以及thresh_ineq来判定样本分到哪个类
    thresh_ineq用于设置是大于thresh为负样本还是小于thresh为负样本"""
    res = np.ones((shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        res[data_mat[:, dimen] <= thresh] = -1.0
    else:
        res[data_mat[:, dimen] > thresh] = -1.0
    return res


# In[ ]:

def build_stump(data_arr, class_labels, D):
    """构建单层决策树，D为权重向量，输入数据和权重，返回最优特征和分割点等信息"""
    data_mat = np.mat(data_arr)
    label_mat = np.mat(class_labels).T
    m, n = shape(data_mat)
    
    num_steps = 10.0    #对每个特征有十个候选thresh
    best_stump = {}    #记录最优特征和最优分割点等信息
    min_error = np.inf
    
    for i in range(n):  #对每个特征
        min_val, max_val = data_mat[:, i].min(), data_mat[:,i].max()
        step_size = (max_val - min_val) / num_steps
        for j in range(-1, int(num_steps) + 1):  #对每个候选值,这里从-1开始意义不大，因为如果当选，则全部样本归为一类
            for inequal in ['lt', 'gt']:  #对每个方向
                thresh = min_val + j * step_size
                predict_res = stump_classify(data_mat, i, thresh, inequal)  #得到预测结果
                
                error_arr = np.mat(np.ones(m, 1))
                error_arr[predict_res == label_mat] = 0  #计算错误率
                weighted_error = D.T * error_arr  #加权后的错误率
                
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh
                    best_stump['ineq'] = inequal
                    best_stump['predict_res'] = predict_res.copy()
    return best_stump, min_error


# In[ ]:

def adaBoost_train(data_arr, class_labels, n_iter):
    """构建adaboost算法，在构建的时候会考虑权重，训练出来的基学习器在预测的时候是不会对样本加权的"""
    weak_class_arr = []  #弱分类器
    m = shape(data_arr)[0]
    weights = np.mat(np.ones(m, 1)) / m  #初始化样本权重，均匀分布
    agg_class_est = np.mat(np.zeros((m, 1)))  # f(x)在训练集上的结果
    
    for i in range(n_iter):
        best_stump, error = build_stump(data_arr, class_labels, weights)  #基分类器
        alpha = 0.5 * np.log((1-error) / max(error, 1e-16))  #防止error为0
        
        best_stump['alpha'] = alpha
        weak_class_arr.append(best_stump)  #记录学习结果
        
        expon = np.multiply(-1 * alpha * np.mat(class_labels).T, best_stump['predict_res'])  # shape: m * 1
        weights = np.multiply(weights, np.exp(expon))  # shape: m * 1
        weights /= weights.sum()  #得到下一轮的权重
        
        agg_class_est += alpha * best_stump['predict_res']
        agg_error = np.multiply(np.sign(agg_class_est) != np.mat(class_labels).T, np.ones((m,1)))  #累计错误率，前者是判断，得到的是true,false，乘以1可以得到数字1，0
        error_rate = agg_error.sum() / m  #平均错误率
        if error_rate == 0:
            break
    return weak_class_arr


# In[ ]:



