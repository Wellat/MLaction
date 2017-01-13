# -*- coding: utf-8 -*-
'''
Tree-Based Regression Methods
'''
from numpy import *

def loadDataSet(fileName):
    '''
    读取一个一tab键为分隔符的文件，然后将每行的内容保存成一组浮点数    
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    '''
    数据集切分函数    
    '''
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]
    return mat0,mat1

def regLeaf(dataSet):
    '''负责生成叶节点'''
    #当chooseBestSplit()函数确定不再对数据进行切分时，将调用本函数来得到叶节点的模型。
    #在回归树中，该模型其实就是目标变量的均值。
    return mean(dataSet[:,-1])

def regErr(dataSet):
    '''
    误差估计函数，该函数在给定的数据上计算目标变量的平方误差，这里直接调用均方差函数
    '''
    return var(dataSet[:,-1]) * shape(dataSet)[0]#返回总方差

def linearSolve(dataSet):
    '''将数据集格式化成目标变量Y和自变量X，X、Y用于执行简单线性回归'''
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#默认最后一列为Y
    xTx = X.T*X
    #若矩阵的逆不存在，抛异常
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)#回归系数
    return ws,X,Y

def modelLeaf(dataSet):
    '''负责生成叶节点模型'''
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    '''误差计算函数'''
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    用最佳方式切分数据集和生成相应的叶节点
    '''  
    #ops为用户指定参数，用于控制函数的停止时机
    tolS = ops[0]; tolN = ops[1]
    #如果所有值相等则退出
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf; bestIndex = 0; bestValue = 0
    #在所有可能的特征及其可能取值上遍历，找到最佳的切分方式
    #最佳切分也就是使得切分后能达到最低误差的切分
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    #如果误差减小不大则退出
    if (S - bestS) < tolS: 
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    #如果切分出的数据集很小则退出
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    #提前终止条件都不满足，返回切分特征和特征值
    return bestIndex,bestValue

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    '''
    树构建函数
    leafType:建立叶节点的函数
    errType:误差计算函数
    ops:包含树构建所需其他参数的元组    
    '''    
    #选择最优的划分特征
    #如果满足停止条件，将返回None和某类模型的值
    #若构建的是回归树，该模型是一个常数；如果是模型树，其模型是一个线性方程
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val #
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    #将数据集分为两份，之后递归调用继续划分
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree  

def isTree(obj):
    '''判断输入变量是否是一棵树'''
    return (type(obj).__name__=='dict')

def getMean(tree):
    '''从上往下遍历树直到叶节点为止，计算它们的平均值'''
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    '''回归树剪枝函数'''
    if shape(testData)[0] == 0: return getMean(tree) #无测试数据则返回树的平均值
    if (isTree(tree['right']) or isTree(tree['left'])):#
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #如果两个分支已经不再是子树，合并它们
    #具体做法是对合并前后的误差进行比较。如果合并后的误差比不合并的误差小就进行合并操作，反之则不合并直接返回
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    #为了和modeTreeEval()保持一致，保留两个输入参数
    return float(model)

def modelTreeEval(model, inDat):
    #对输入数据进行格式化处理，在原数据矩阵上增加第0列，元素的值都是1
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    '''
    # 在给定树结构的情况下，对于单个数据点，该函数会给出一个预测值。
    # modeEval是对叶节点进行预测的函数引用，指定树的类型，以便在叶节点上调用合适的模型。
    # 此函数自顶向下遍历整棵树，直到命中叶节点为止，一旦到达叶节点，它就会在输入数据上
    # 调用modelEval()函数，该函数的默认值为regTreeEval()    
    '''
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    # 多次调用treeForeCast()函数，以向量形式返回预测值，在整个测试集进行预测非常有用
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat
    
if __name__ == "__main__":
    from numpy import *
#    testMat = mat(eye(6))
#    mat0,mat1 = binSplitDataSet(testMat,1,0.5)
    
    myDat=loadDataSet('ex00.txt')
    myMat=mat(myDat)
    retTree=createTree(myMat)    
    
    


    
    