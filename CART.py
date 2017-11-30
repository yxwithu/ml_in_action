
# coding: utf-8

# cart回归树的构建

# In[ ]:

import numpy as np


# In[ ]:

def reg_leaf(dataSet):
    """生成叶子节点，返回这个叶子上样本标记的平均值"""
    return np.mean(dataSet[:, -1])

def reg_err(dataSet):
    """计算平方误差"""
    return np.var(dataSet[:, -1]) * shape(dataSet)[0]


# In[ ]:

def bin_split_data(dataSet, feat_index, split_value):
    arr0 = []
    arr1 = []
    for i in range(shape(dataSet)[0]):
        if dataSet[i, feat_index] <= split_value:
            arr0.append(dataSet[i])
        else:
            arr1.append(dataSet[i])
    return np.mat(arr0), np.mat(arr1)


# In[ ]:

def choose_best_split(dataSet, leaf_type=reg_leaf, err_type = reg_err, ops=(1,4)):
    """选择最优分裂节点和分裂值"""
    tolS = ops[0]  #误差减少阈值，达到tolS才允许分裂
    tolN = ops[1]  #最少分割样本，达到tolN才允许分裂
    
    if len(set(dataSet[:,-1])) == 1 or len(dataSet) <= tolN:  #样本的值相等，没必要分割了
        return None, leaf_type(dataSet)
    
    m, n = shape(dataSet)
    ori_err = reg_err(dataSet)
    
    lowest_err = np.inf
    best_index = -1
    best_value = -1
    
    for index in range(n - 1):
        for value in set(dataSet[:, index]):
            mat0, mat1 = bin_split_data(dataSet, index, value)
            if shape(mat0)[0] < tolN or shape(mat1) < tolN:  #子节点样本数过少
                continue
            new_err = reg_err(mat0, mat1)  #子树的平方误差和
            if new_err < lowest_err:
                lowest_err = new_err
                best_index = index
                best_value = value
    
    if best_index == -1 or (ori_err - lowest_err) < tolS:  #误差减少太小
        return None, leaf_type(dataSet)
    
    return best_index, best_value

