
# coding: utf-8

# In[ ]:

import numpy as np
import random


# In[ ]:

class opt_struct:
    def __init__(self, X_train, y_train, C, tol):
        self.X = X_train
        self.labels = y_train
        self.C = C  
        self.tol = tol
        self.m = shape(X_train)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))  #要求的alpha值
        self.b = 0  #偏移量
        self.e_cache = np.mat(np.zeros((self.m, 2)))  #E的缓存，第一列是否有效（是否计算好了），第二列为计算好的值
        
def calc_ek(oS, k):
    """利用已有的alpha, x, y, 计算第k个样本的预测值与实际值的差距"""
    fxk = np.multiply(oS.alphas, oS.labels).transpose() * (os.X * os.X[k, :].T) + oS.b  #利用公式，求出f(xi)
    return fxk - oS.labels[k]  #与实际label的差别

def select_inner_random(i, m):
    """随机选取一个alpha j与alpha i配对"""
    j = i
    while(j == i):
        j = int(random.uniform(0, m))
    return j

def select_inner(i, oS, ei):
    """在外层为alpha_i的情况下选择内层alpha_j。
    计算每个j的预测值与真实值的差距，选择差距与ei相差最多的那个j，
    因为这样alphaj变化最大，使目标函数有足够的下降"""
    max_j = -1
    choose_ej = 0
    max_delta_e = 0

    oS.e_cache[i] = [1, ei]
    valid_ecache_list = np.nonzero(oS.e_cache[:, 0].A)[0]  #得到有效的缓存index
    if len(valid_ecache_list) > 1:   #有有效的缓存
        for j in valid_ecache_list:
            if j == i:
                continue
            ej = calc_ek(oS, j)
            delta_e = abs(ei - ej)
            if delta_e > max_delta_e:
                max_delta_e = delta_e
                max_j = j
                choose_ej = ej
    if max_j == -1:   #第一次循环没有有效缓存，随机选取一个
        max_j = select_inner_random(i)
        choose_ej = calc_ek(oS, max_j)
    return max_j, choose_ej

def update_ek(oS, k):
    """用于alpha值优化以后"""
    oS.e_cache[k] = [1, calc_ek(oS, k)]
    
def clip_alpha(alpha_new, L, H):
    """在L和H限制下的，alpha_new能取到的值，用于SMO求解"""
    if H < alpha_new:
        return H
    if L > alpha_new:
        return L
    return alpha_new


# In[ ]:

def inner(i, oS):
    """找内层alpha，首先外层的alpha需要达到破坏KKT条件到一定容忍度的程度，
    选择内部节点找|E1 - E2|最大的，因为这样会使目标函数优化最快,
    有几个边界条件：
    1. 上下界相同，没有优化空间
    2. 核化后的样本的差的平方小于等于0
    3. 更新后的alph2变化不大（也就不能给目标函数带来很大的变化）
    """
    ei = calc_ek(oS, i)  #计算外层误差
    alpha_i = oS.alpha[i]
    if ((alpha_i < oS.C and ei * oS.labels[i] < -oS.tol) or 
       (alpha_i > 0 and ei * oS.labels[i] > oS.tol)):  #违反kkt，符合时第一种情况应该是>=0的，第二种情况应该<=0
        j, ej = select_inner(i, oS, ei)  #选择内部节点
        alpha_1_old = oS.alpha[i].copy()
        alpha_2_old = oS.alpha[j].copy()
        if oS.labels[i] != oS.labels[j]:
            L = max(0, alpha_2_old - alpha_1_old)
            H = min(oS.C, oS.C + alpha_2_old - alpha_1_old)
        else:
            L = max(0, alpha_2_old + alpha_1_old - oS.C)
            H = min(oS.C, alpha_2_old + alpha_1_old)
        if L == H:  #相等则没有优化的空间
            print("L == H")
            return 0
        eta = oS.X[i,:] * oS.X[i,:].T + oS.X[j,:] * oS.X[j,:].T - 2.0 * oS.X[i,:] * oS.X[j,:].T 
        if eta <= 0:  #这个值是核化后两个样本差的平方，不可能小于0，同时这个值是要做分母的，不能等于0
            print("eat <= 0")
            return 0
        oS.alpha[j] += oS.labels[j] * (ei - ej) / eta  # alpha 2 new unc
        oS.alpha[j] = clip_alpha(oS.alpha[j], L, H)  #取限制范围内能取得的最优值
        update_ek(oS, j)  #更新值
        if (abs(oS.alpha[j] - alpha_2_old) < 0.00001):
            print("j not moving enough")  #j不能带来足够的变化
            return 0
        oS.alpha[i] += oS.labels[i] * oS.labels[j] * (alpha_2_old - oS.alpha[j])  #更新alpha 1
        update_ek(oS, i)
        b1 = oS.b - ei - oS.labels[i] * (oS.alpha[i] - alpha_1_old) * oS.X[i,:] * oS.X[i,:].T - oS.labels[j] * oS.X[i,:] * oS.X[j,:].T *(oS.alpha[j] - alpha_2_old)
        b2 = oS.b - ej - oS.labels[i] * (oS.alpha[i] - alpha_1_old) * oS.X[i,:] * oS.X[j,:].T - oS.labels[j] * oS.X[j,:] * oS.X[j,:].T *(oS.alpha[j] - alpha_2_old)
        if oS.alpha[i] > 0 and oS.alpha[i] < oS.C:
            oS.b = b1
        elif oS.alpha[j] > 0 and oS.alpha[j] < oS.C:
            oS.b = b2
        else:
            os.b = (b1 + b2) / 2
        return 1
    else:
        return 0


# In[ ]:

def smo(data_mat, class_labels, C, toler, n_iter, kTup=('lin', 0)):
    """SMO主函数，先选择外层alpha，再选择内层alpha
    初始化时所有alpha均为0，没有边界alpha，所以对全集寻找
    后面对边界alpha进行寻找，如果边界alpha没有找到，再进行全集寻找
    """
    oS = opt_struct(np.mat(data_mat), np.mat(class_labels).transpose(), C, toler)
    cur_iter = 0
    entire_set = True
    alpha_pair_changed = 0
    
    while(cur_iter < n_iter and (alpha_pair_changed > 0 or entire_set)):
        alpha_pair_changed = 0
        if entire_set:
            for i in range(oS.m):
                alpha_pair_changed += inner(i, oS)
            cur_iter += 1
        else:
            non_bounds = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in non_bounds:
                alpha_pair_changed += inner(i, oS)
            cur_iter += 1
        if entire_set:
            entire_set = False
        elif alpha_pair_changed == 0:
            entire_set = True
    return oS.b, oS.alphas


# In[ ]:



