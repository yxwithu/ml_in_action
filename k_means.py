
# coding: utf-8

# In[ ]:

import numpy as np
import random


# In[ ]:

def calc_dist(vec1, vec2):
    """计算两个向量的欧氏距离"""
    return np.sqrt(sum(np.power(vec1-vec2, 2)))

def rand_cent(dataSet, k):
    """取k个随机质心，质心的每个特征的值是在数据上下界中随机选取
    不一定是数据中包含的值"""
    n = shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_val = min(dataSet[:, j])
        range_val = max(dataSet[:, j]) - min_val
        centroids[j] = min_val + range_val * random.rand(k, 1)
    return centroids


# In[ ]:

def Kmeans(dataSet, k, dist_eval = calc_dist, create_cent = rand_cent):
    """基本Kmeans算法，初始化质心->分配->重新计算质心->直到所有点的所属簇都不再改变"""
    m = shape(dataSet)[0]
    cluster_assign = np.mat(np.zeros((m, 2)))  #样本属于哪个簇和距离
    centroids = create_cent(dataSet, k)  #随机创建质心
    
    cluster_change = True  #程序终止条件
    while cluster_change:
        cluster_change = False
        
        #更新每个样本的所在簇
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist = dist_eval(dataSet[i, :], centroids[j, :])
                if dist < min_dist:
                    min_dist = dist
                    min_index = j
            if min_index != cluster_assign[i]:
                cluster_change = True
                cluster_assign[i] = [min_index, min_dist]
        
        if not cluster_change:
            break
        
        #更新每个簇
        for cent in range(k):
            cluster_sample = dataSet[np.nonzero(cluster_assign[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(cluster_sample, axis = 0)
    
    return centroids, cluster_assign


# 二分K_means算法

# In[ ]:

def bin_Kmeans(dataSet, k, dist_eval = calc_dist):
    """
    Kmeans容易收敛到局部最小值，为克服，有二分Kmeans:
    1. 将所有点看成一个簇
    2. 当簇数目小于k时继续划分：
           对于每一个簇:
               计算总误差
               在给定的簇上面进行2Means聚类
               计算将该簇一分为2之后的误差 + 其他簇类的误差作为新的总误差
           选择使得总误差最小的的分割
    """
    m = shape(dataSet)[0]
    cluster_assign = np.mat(np.zeros((m, 2)))  #第一列保存所属簇id， 第二列保存距离
    
    centroid0 = np.mean(dataSet, axis = 0).tolist()[0]  #初始簇为全数据集的中心
    cent_list = [centroid0]
    
    for i in range(m):
        cluster_assign[i, 1] = dist_eval(np.mat(centroid0), dataSet[i, :]) ** 2  #用欧式距离的平方，更重视那些远离中心的点
        
    while len(cent_list) < k:
        lowest_sse = np.inf  # sum of squared error
        
        #选择最优分割簇，使总误差最小
        for i in range(len(cent_list)):  
            cluster_samples = dataSet[np.nonzero(cluster_assign[:, 0].A == i)[0], :]  #这个簇的所有样本
            centroids, clusters = Kmeans(cluster_samples, 2, dist_eval)  #对这个簇的样本一分为2
            sse_other_cluster = sum(cluster_assign[np.nonzero(cluster_assign[:, 0].A != i)[0], 1])  #其他簇类的误差
            sse_split = sum(clusters[:, 1])  #划分部分的误差
            
            if sse_split + sse_other_cluster < lowest_sse:
                lowest_sse = sse_split + sse_other_cluster
                best_new_cnets = centroids
                best_clusters = clusters
                best_split_cent = i
        
        #更新簇的分配结果
        best_clusters[np.nonzero(best_clusters[:, 0].A == 1)[0], 0] = len(cent_list)  #新的簇id
        best_clusters[np.nonzero(best_clusters[:, 0].A == 0)[0], 0] = best_split_cent  #被分割的簇id
        
        cent_list[best_split_cent] = best_new_cnets[0:]  #更新簇的特征值
        cent_list.append(best_new_cnets[1:])
        
        cluster_assign[np.nonzero(cluster_assign[:, 0].A == best_split_cent)[0], :] = best_clusters  #更新总的样本所属簇记录
        
    return np.mat(cent_list), cluster_assign


# In[ ]:



