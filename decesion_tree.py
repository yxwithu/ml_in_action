
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

from math import log


# In[3]:

def calc_ent(dataSet):
    """计算数据集的香农熵"""
    sample_cnt = len(dataSet)
    label_cnt_dict ={}
    for feat in dataSet:
        label = feat[-1]
        label_cnt_dict[label] = label_cnt_dict.get(label, 0) + 1
    ent = 0
    for key, value in label_cnt_dict.items():
        prob = value / sample_cnt
        ent-= prob * log(prob, 2)
    return ent


# In[4]:

def splitDataSet(dataSet, col_id):
    """将数据集按某一列特征分割"""
    feat_data_dict = {}
    for i in range(len(dataSet)):
        feat = dataSet[i][col_id]
        if feat not in feat_data_dict:
            feat_data_dict[feat] = []
        data = dataSet[i][:col_id] + dataSet[i][col_id+1:]
        feat_data_dict[feat].append(data)
    return feat_data_dict


# In[5]:

def choose_best_feat_ent_ratio(dataSet):
    """结合信息增益和信息增益率：在信息增益超过平均值的特征选择信息增益率最高的"""
    gain_dict = {}
    ratio_dict = {}
    ori_ent = calc_ent(dataSet)
    sample_cnt = len(dataSet)
    
    for col_id in range(len(dataSet[0]) - 1):
        feat_data_dict = splitDataSet(dataSet, col_id)
        cur_gain = ori_ent
        cur_iv = 0
        for key, data in feat_data_dict.items():
            prob = len(data) / sample_cnt
            cur_iv -= prob * log(prob, 2)  #累加IV(feat)
            sub_ent = calc_ent(data)
            cur_gain -= prob * sub_ent  #累积增益
            
        gain_dict[col_id] = cur_gain
        ratio_dict[col_id] = cur_gain / cur_iv
    
    mean_gain = np.mean(list(gain_dict.values()))
    best_col = 0
    max_ratio = 0
    for col_id in gain_dict:
        if gain_dict[col_id] > mean_gain and ratio_dict[col_id] > max_ratio:
            max_ratio = ratio_dict[col_id]
            best_col = col_id
    return best_col


# In[6]:

def choose_best_feature_ent_gain(dataSet):
    """选择最好的分割特征"""
    max_gain = 0
    ori_ent = calc_ent(dataSet)
    best_col = 0
    sample_cnt = len(dataSet)
    
    for col_id in range(len(dataSet[0]) - 1):
        feat_data_dict = splitDataSet(dataSet, col_id)
        cur_gain = ori_ent
        for key, data in feat_data_dict.items():
            prob = len(data) / sample_cnt  #子树数据量占比
            sub_ent = calc_ent(data)   #子树信息熵
            cur_gain -= prob * sub_ent   #增益
        if cur_gain > max_gain:
            max_gain = cur_gain
            best_col = col_id
    return best_col


# In[7]:

def get_majority_cnt_label(label_list):
    """得到列表中出现次数最多的label，主要用于决定叶子节点的label"""
    cnt_dict = {}
    max_cnt = 0
    max_label = 0
    for label in label_list:
        cnt = cnt_dict.get(label, 0) + 1
        cnt_dict[label] = cnt
        if cnt > max_cnt:
            max_cnt = cnt
            max_label = label
    return label


# In[8]:

def create_tree(dataSet, feat_names):
    """递归创建一棵ID3决策树"""
    y_train = [row[-1] for row in dataSet]
    if len(set(y_train)) == 1:  #类别完全相同，停止继续划分，返回类别
        return y_train[0]
    if len(dataSet[0]) == 1:  #没有特征可以划分了，直接返回最多的特征
        return get_majority_cnt_label(y_train)
    
    best_feat = choose_best_feat_ent_ratio(dataSet)  #找到最优分割特征
    best_feat_name = feat_names[best_feat]
    
    myTree = {best_feat_name:{}}  #开始构建二叉树
    del feat_names[best_feat]
    feat_data_dict = splitDataSet(col_id=best_feat, dataSet=dataSet)
    for feat_value, data in feat_data_dict.items():
        sub_feat_names = feat_names[:]  #拷贝赋值，防止被修改
        myTree[best_feat_name][feat_value] = create_tree(data, sub_feat_names)   #保证传进去的不是空的数据集
    return myTree


# In[9]:

def create_data_set():
    dataSet = [[1,1,'yes'],
              [1,0,'no'],
              [0,1, 'no'],
              [0,1, 'no']]
    feat_names = ['no surfacing', 'flippers']
    return dataSet, feat_names


# In[10]:

dataSet, feat_names = create_data_set()
create_tree(dataSet, feat_names)


# In[ ]:



