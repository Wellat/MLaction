# -*-coding:utf-8 -*-
'''
Logistic Regression Working Module
'''
from numpy import *

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        #为方便计算将x0设为1.0
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    梯度上升算法
    dataMatIn:2维NumPy数组 (100x3)
    classLabels:类标签 (1x100)
    '''
    #将输入转换为NumPy矩阵的数据类型
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose() 
    m,n = shape(dataMatrix)
    #向目标移动的步长
    alpha = 0.001
    #迭代次数
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights

def plotBestFit(dataMat,labelMat,weights):
    '''
    画出数据集和Logistic回归最佳拟合直线的函数
    '''
    import matplotlib.pyplot as plt
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    #根据类别分别保存点
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    #此处设置了Sigmoid的z为0，因为0是两个分类的分界处
    #即：0=w0x0+w1x1+w2x2
    #注意：x0=1,x1=x,解出x2=y
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y.transpose())
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=500):
    '''
    改进的随机梯度算法
    '''
    dataMatrix = array(dataMatrix)
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            #alpha会随着迭代次数不断减小，但存在常数项，它不会小到0
            #这种设置可以缓解数据波动
            alpha = 4/(1.0+j+i)+0.0001
            #通过随机选取样本来更新回归系数
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

def multiTest():
    '''
    多次调用colicTest()函数，求结果的平均值
    '''
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))

def classifyVector(inX, weights):
    '''
    分类函数    
    '''
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

if __name__ == "__main__":
#    dataMat,labelMat = loadDataSet()
#    weights = gradAscent(dataMat,labelMat)
#    weights = stocGradAscent1(dataMat,labelMat)
#    plotBestFit(dataMat,labelMat,weights)

    '''
    从疝气病症预测病马的死亡率
    '''
#    colicTest()
    multiTest()
    


