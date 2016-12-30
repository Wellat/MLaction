# -*- coding: utf-8 -*-
'''
Adaboost is short for Adaptive Boosting
'''
from numpy import *

def loadSimpData():
    '''
    输入简单测试数据    
    '''
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    '''读取数据函数'''
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    '''
    通过阈值比较对数据进行分类    
    '''
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    '''
    建立一个单层决策树
    输人为权重向量D，
    返回具有最小错误率的单层决策树、最小的错误率以及估计的类别向量
    '''    
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #
    for i in range(n):#对数据集中的每一个特征
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):#对每个步长
            for inequal in ['lt', 'gt']: #对每个不等号
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #计算加权错误率
                #print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树                
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    基于单层决策树的AdaBoost训练过程
    '''
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #初始化权重向量为1/m
    aggClassEst = mat(zeros((m,1)))#记录每个数据点的类别估计累计值
    for i in range(numIt):
        #建立一个单层决策树
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
#        print("D:",D.T)
        #计算alpha，此处分母用max(error,1e-16)以防止error=0
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)
#        print("classEst: ",classEst.T)
        #计算下一次迭代的D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))                              
        D = D/D.sum()
        #以下计算训练错误率，如果总错误率为0，则终止循环
        aggClassEst += alpha*classEst
#        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print("total error: ",errorRate)
        if errorRate == 0.0: break
    return weakClassArr,aggClassEst

def adaClassify(datToClass,classifierArr):
    '''
    利用训练出的多个弱分类器进行分类    
    datToClass:待分类数据
    classifierArr:训练的结果
    '''
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    #遍历classifierArr中的所有弱分类器，并基于stumpClassify对每个分类器得到一个类别的估计值
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
#        print(aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep;
        else:
            delX = xStep; delY = 0;
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print("the Area Under the Curve is: ",ySum*xStep)


if __name__ == "__main__":
#    datMat,classLabels = loadSimpData()
#    D = mat(ones((5,1))/5)
#    bestStump,minError,bestClasEst = buildStump(datMat,classLabels,D)
#    weakClassArr,aggClassEst = adaBoostTrainDS(datMat,classLabels,numIt=40)
#    adaClassify([0,0],weakClassArr)
    
    '''马疝病测试'''
    #导入训练数据
    datArr,labelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr,aggClassEst = adaBoostTrainDS(datArr,labelArr,50)
    
    #导入测试数据
    testArr,testLabelArr = loadDataSet('horseColicTest2.txt')
    prediction = adaClassify(testArr,weakClassArr) 
    #计算错误率
    errArr = mat(ones((67,1)))
    errArr[prediction != mat(testLabelArr).T].sum()/67
    
    
    