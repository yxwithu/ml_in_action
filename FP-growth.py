
# coding: utf-8

# In[1]:

class treeNode:
    def __init__(self, name, cnt, parent):
        self.name = name  #item的名称
        self.cnt = cnt  #出现次数
        self.linkNode = None  #相同名称的但是不同路径的item
        self.parent = parent  #父节点
        self.children = {}  #子节点，可以有多个
        
    def inc(self, num_occur):
        """又出现了几次"""
        self.cnt += num_occur
        
    def show_info(self, level = 1):
        """分层打印信息"""
        print("  " * level, self.name, ' ', self.cnt)
        for child in self.children.values():
            child.show_info(level + 1)


# In[2]:

def createTree(dataSet, min_sup = 1):
    """
    输入dataSet为{事务:cnt}, min_sup为支持度阈值
    
    1. 先遍历数据集，得到每个元素项的出现频率，放在头指针表中,头指针表结构： {name : [count, pointer]}
    2. 过滤掉低于支持度阈值的元素项
    3. 对每个事务（数据）中的项按照出现频率排序，即路径
    4. 按路径创建树，沿路节点累加出现次数，并更新头指针表
    
    返回头指针表和FPTree头结点
    """
    header_table = {}  #头指针表，记录每个项集出现的次数
    for trans in dataSet:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + 1
    
    key_set = set(header_table.keys())
    for key in key_set:  #过滤掉低于支持度阈值的元素项
        if header_table[key] < min_sup:
            del header_table[key]
            
    if len(header_table.keys()) == 0:
        return None, None
    
    for key in header_table:
        header_table[key] = [header_table[key], None]  #头指针表结构： {name : [count, pointer]}
     
    root = treeNode('null set', 1, None)  #初始化根节点
    for trans, cnt in dataSet.items():  #对每一条事务都进行更新
        localD = {}
        for item in trans:
            if item in header_table:  #是频繁项集
                localD[item] = header_table[item][0]
        if len(localD) > 0:
            orderedItems = sorted(localD, key=localD.get, reverse=True)  #按出现频次降序排序
            updateTree(orderedItems, root, header_table, cnt)  #更新FP-Tree
    return root, header_table


# In[3]:

def updateTree(items, root, header_table, cnt):
    """根据cnt条新的排序好的事务对FP_TREE和header_table进行更新"""
    if items[0] in root.children:
        root.children[items[0]].inc(cnt)  #已有children记录，直接加上计数
    else:
        root.children[items[0]] = treeNode(items[0], cnt, root)  #创建新的结点
        if header_table[items[0]][1] == None:  #更新header_table的pointer，有可能是None,有可能不是
            header_table[items[0]][1] = root.children[items[0]]
        else:  #说明开辟了一条新的路径
            node = header_table[items[0]][1]  #从header_table开始找
            while(node.linkNode != None):
                node = node.linkNode
            node.linkNode = root.children[items[0]]
    if len(items) > 1:
        updateTree(items[1:], root.children[items[0]], header_table, cnt)


# In[4]:

def findPrefixPath(treeNode):
    """发现以给定元素项结尾的所有路径，路径上结点的频次"""
    condaPats = {}  # {path : count}
    while treeNode != None:
        prefixPath = []  #路径
        node = treeNode
        while node.parent != None:
            prefixPath.append(node.name)
            node = node.parent
        if len(prefixPath) > 1:
            condaPats[frozenset(prefixPath[1:])] = treeNode.cnt  #都更新为treeNode的计数
        treeNode = treeNode.linkNode
    return condaPats


# In[6]:

def minTree(root, header_table, min_sup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(header_table.items(), key = lambda p: p[1][0])]  #对dict中value为list的排序
    
    for basePat in bigL:  #从头指针表的底端开始
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(header_table[basePat][1])
        condTree, condHead = createTree(condPattBases, min_sup)
        if condHead != None:
            minTree(condTree, condHead, min_sup, newFreqSet, freqItemList)

