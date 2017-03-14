# -*- coding: utf-8 -*-
'''
Decision Tree Source Code for Machine Learning in Action Ch. 3
'''
from math import log
import operator

def createDataSet():
    '''
    产生测试数据
    '''
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    计算给定数据集的香农熵
    '''
    numEntries = len(dataSet)
    labelCounts = {}
    #统计每个类别出现的次数，保存在字典labelCounts中
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 #如果当前键值不存在，则扩展字典并将当前键值加入字典
    shannonEnt = 0.0
    for key in labelCounts:
        #使用所有类标签的发生频率计算类别出现的概率
        prob = float(labelCounts[key])/numEntries
        #用这个概率计算香农熵
        shannonEnt -= prob * log(prob,2) #取2为底的对数
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    '''
    按照给定特征划分数据集
    dataSet：待划分的数据集
    axis：   划分数据集的第axis个特征
    value：  特征的返回值（比较值）
    '''
    retDataSet = []
    #遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
            #extend()和append()方法功能相似，但在处理列表时，处理结果完全不同
            #a=[1,2,3]  b=[4,5,6]
            #a.append(b) = [1,2,3,[4,5,6]]
            #a.extend(b) = [1,2,3,4,5,6]
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    '''
    选择最好的数据集划分方式
    输入：数据集
    输出：最优分类的特征的index
    '''
    #计算特征数量
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        #创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        #计算每种划分方式的信息熵
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy
        #计算最好的信息增益，即infoGain越大划分效果越好
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    '''
    投票表决函数
    输入classList:标签集合，本例为：['yes', 'yes', 'no', 'no', 'no']
    输出：得票数最多的分类名称
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #把分类结果进行排序，然后返回得票数最多的分类结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    '''
    创建树
    输入：数据集和标签列表
    输出：树的所有信息
    '''
    # classList为数据集的所有类标签
    classList = [example[-1] for example in dataSet]
    # 停止条件1:所有类标签完全相同，直接返回该类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    # 停止条件2:遍历完所有特征时仍不能将数据集划分成仅包含唯一类别的分组，则
    # 返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优分类特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # myTree存储树的所有信息
    myTree = {bestFeatLabel:{}}
    # 以下得到列表包含的所有属性值
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历当前选择特征包含的所有属性值
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                         
    
def classify(inputTree,featLabels,testVec):
    '''
    决策树的分类函数
    inputTree:训练好的树信息
    featLabels:标签列表
    testVec:测试向量
    '''
    # 在2.7中，找到key所对应的第一个元素为：firstStr = myTree.keys()[0]，
    # 这在3.4中运行会报错：‘dict_keys‘ object does not support indexing，这是因为python3改变了dict.keys,
    # 返回的是dict_keys对象,支持iterable 但不支持indexable，
    # 我们可以将其明确的转化成list，则此项功能在3中应这样实现：
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]
    secondDict = inputTree[firstStr]
    # 将标签字符串转换成索引
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 递归遍历整棵树，比较testVec变量中的值与树节点的值，如果到达叶子节点，则返回当前节点的分类标签
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    '''
    使用pickle模块存储决策树
    '''
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    '''
    导入决策树模型
    '''
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
    
if __name__== "__main__":  
    '''
    计算给定数据集的香农熵
    '''
#    dataSet,labels = createDataSet()
#    shannonEnt = calcShannonEnt(dataSet)
    '''
    按照给定特征划分数据集
    '''
    ans = splitDataSet(dataSet,1,1)
    '''
    选择最好的数据集划分方式
    '''
#    ans = chooseBestFeatureToSplit(dataSet)
#    ans = majorityCnt(classList)
#    myTree = createTree(dataSet,labels)
    '''
    测试分类效果
    '''
#    dataSet,labels = createDataSet()
#    myTree = createTree(dataSet,labels)
#    ans = classify(myTree,labels,[1,0])
    '''
    存取操作
    '''
#    storeTree(myTree,'mt.txt')
#    myTree2 = grabTree('mt.txt')
    
    
