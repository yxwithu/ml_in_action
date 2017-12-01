
# coding: utf-8

# In[ ]:

import numpy as np


# In[ ]:

def load_dataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def create_can_set1(dataSet):
    """得到单元素项集, 项集后面会作为Key值，所以用forzenset"""
    can_set = set()
    can_lst = []
    for lst in dataSet:
        for can in lst:
            if not can in can_set:
                can_set.add(can)
                can_lst.append([can])
    return list(map(frozenset, can_lst))  #将canlst中的每一个list都转化成forzenset


# In[ ]:

def scan_data(dataSet, can_set, min_support):
    """得到达到最小支持度的频繁项集和对应的支持度"""
    can_cnt = {}
    for lst in dataSet:
        for can in can_set:
            if can.issubset(lst):  #这个项集出现在集合中
                can_cnt[can] = can_cnt.get(can, 0) + 1
    total = len(dataSet)
    ret_lst = []  #达到最小支持度的频繁项集
    support_data = {}  #达到最小支持度的频繁项集的支持度
    
    for key, value in can_cnt.items():
        support = value / total
        if support >= min_support:
            ret_lst.append(key)
            support_data[key] = support
    
    return ret_lst, support_data


# In[ ]:

def apriori_gen(lk, k):
    """lk为频繁项集列表，k为每个项集元素个数,函数用于生成下一轮k+1个元素的频繁项集"""
    ret_lst = []  #下一轮结果
    len_lk = len(lk)
    for i in range(len_lk):
        l1 = list(lk[i])[:k-1]
        l1.sort()
        for j in range(i+1, len_lk):
            l2 = list(lk[j])[:k-1]
            l2.sort()
            if l1 == l2:
                ret_lst.append(lk[i] | lk[j])  #只需要比较前k-2个元素是否相同，相同则组合即可
    return ret_lst


# In[ ]:

def apriori(dataSet, min_support = 0.5):
    """apriori算法挖掘频繁项集函数，生成并记录每一轮的频繁项集和他们的支持度
    """
    can_set = create_can_set1(dataSet)
    data_set = list(map(set, dataSet))  #将数据集里的列表转换成set，python2不用加list
    can_set1, support_data = scan_data(data_set, can_set, min_support)  #第一轮频繁项集
    
    can_sets = [can_set1]  #L用于记录所有生成的频繁项集 [[第一轮],[第二轮],...]
    
    k = 2
    while len(can_sets[k-2]) > 0:  #第k-1轮频繁项集不为空
        can_setk = apriori_gen(can_sets[k-2], k-1)  #用k-1轮，存放在k-2位置上，生成k轮
        can_setk, support_k = scan_data(data_set, can_setk, min_support)  #过滤支持度过低的
        support_data.update(support_k)  #将第k轮的结果放入到support_data中
        can_sets.append(can_setk)
        k += 1
    return can_sets, support_data


# In[ ]:

def generate_rules(L, support_data, min_conf = 0.5):
    big_rule_list = []
    sub_set_list = []
    for i in range(0, len(L)):
        for freq_set in L[i]:
            for sub_set in sub_set_list:
                if sub_set.issubset(freq_set):
                    conf = support_data[freq_set] / support_data[freq_set - sub_set]
                    big_rule = (freq_set - sub_set, sub_set, conf)
                    if conf >= min_conf and big_rule not in big_rule_list:
                        # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                        big_rule_list.append(big_rule)
            sub_set_list.append(freq_set)
    return big_rule_list


# In[ ]:



