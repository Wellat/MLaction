# -*- coding: utf-8 -*-

from numpy import *

def loadDataSet(fileName):
    '''导入数据'''
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    '''求标准回归系数'''
    xMat = mat(xArr); yMat = mat(yArr).T
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:#判断行列式是否为0
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T*yMat)#也可以用NumPy库的函数求解：ws=linalg.solve(xTx,xMat.T*yMatT)
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    '''局部加权线性回归函数'''
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))#创建对角矩阵
    for j in range(m):        
        diffMat = testPoint - xMat[j,:]
        #高斯核计算权重
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    '''为数据集中每个点调用lwlr()'''
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

def lwlrTestPlot(xArr,yArr,k=1.0):  #same thing as lwlrTest except it sorts X first
    yHat = zeros(shape(yArr))       #easier for plotting
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i] = lwlr(xCopy[i],xArr,yArr,k)
    return yHat,xCopy

def ridgeRegres(xMat,yMat,lam=0.2):
    '''计算岭回归系数'''
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
def ridgeTest(xArr,yArr):
    '''用于在一组lambda上测试结果'''
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean     #数据标准化
    xMeans = mean(xMat,0)   
    xVar = var(xMat,0)      
    xMat = (xMat - xMeans)/xVar #所有特征减去各自的均值并除以方差
    numTestPts = 30 #取30个不同的lambda调用函数
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def regularize(xMat):
    '''数据标准化函数'''
    inMat = xMat.copy()
    inMeans = mean(inMat,0)
    inVar = var(inMat,0)
    inMat = (inMat - inMeans)/inVar
    return inMat
    
def rssError(yArr,yHatArr): 
    '''计算均方误差大小'''
    return ((yArr-yHatArr)**2).sum()

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    '''
    逐步线性回归算法
    eps：表示每次迭代需要调整的步长
    '''
    xMat = mat(xArr); yMat=mat(yArr).T
    yMean = mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n=shape(xMat)
    returnMat = zeros((numIt,n)) #testing code remove
    #为了实现贪心算法建立ws的两份副本
    ws = zeros((n,1)); wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):#对每个特征
            for sign in [-1,1]:#分别计算增加或减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)
                #取最小误差
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:]=ws.T
    return returnMat

def scrapePage(inFile,outFile,yr,numPce,origPrc):
    from bs4 import BeautifulSoup
    fr = open(inFile,'r',encoding= 'utf8'); fw=open(outFile,'a') #a is append mode writing
    soup = BeautifulSoup(fr.read())
    i=1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
            print("%s\t%d\t%s" % (priceStr,newFlag,title))
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()
    
#from time import sleep
#import json
#import urllib.request 
#def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
#    sleep(2)
#    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
#    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
#    pg = urllib.request.urlopen(searchURL)
#    retDict = json.loads(pg.read())
#    for i in range(len(retDict['items'])):
#        try:
#            currItem = retDict['items'][i]
#            if currItem['product']['condition'] == 'new':
#                newFlag = 1
#            else: newFlag = 0
#            listOfInv = currItem['product']['inventories']
#            for item in listOfInv:
#                sellingPrice = item['price']
#                if  sellingPrice > origPrc * 0.5:
#                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
#                    retX.append([yr, numPce, newFlag, origPrc])
#                    retY.append(sellingPrice)
#        except: print('problem with item %d' % i)
    
def setDataCollect(retX=[], retY=[]):
    '''数据获取方式一（不可用）'''
#    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
#    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
#    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
#    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
#    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
#    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)
    '''数据获取方式二'''
    scrapePage("setHtml/lego8288.html","data/lego8288.txt",2006, 800, 49.99)
    scrapePage("setHtml/lego10030.html","data/lego10030.txt", 2002, 3096, 269.99)
    scrapePage("setHtml/lego10179.html","data/lego10179.txt", 2007, 5195, 499.99)
    scrapePage("setHtml/lego10181.html","data/lego10181.txt", 2007, 3428, 199.99)
    scrapePage("setHtml/lego10189.html","data/lego10189.txt", 2008, 5922, 299.99)
    scrapePage("setHtml/lego10196.html","data/lego10196.txt", 2009, 3263, 249.99)
    
def crossValidation(xArr,yArr,numVal=10):
    '''
    交叉验证测试岭回归
    numVal:交叉验证次数
    '''    
    m = len(yArr)                           
    indexList = list(range(m))
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX=[]; trainY=[]
        testX = []; testY = []
        random.shuffle(indexList)#打乱顺序
        for j in range(m):#构建训练和测试数据，10%用于测试
            if j < m*0.9: 
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY)    #30组不同参数下的回归系数集
        for k in range(30):#遍历30个回归系数集
            matTestX = mat(testX); matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)
            varTrain = var(matTrainX,0)
            matTestX = (matTestX-meanTrain)/varTrain #用训练参数标准化测试数据
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#预测值
            errorMat[i,k]=rssError(yEst.T.A,array(testY))#计算预测平方误差
#            print(errorMat[i,k])
    #在完成所有交叉验证后，errorMat保存了ridgeTest()每个lambda对应的多个误差值
    meanErrors = mean(errorMat,0)#计算每组平均误差
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors==minMean)]#平均误差最小的组的回归系数即为所求最佳
    #岭回归使用了数据标准化，而strandRegres()则没有，因此为了将上述比较可视化还需将数据还原    
    xMat = mat(xArr); yMat=mat(yArr).T
    meanX = mean(xMat,0); varX = var(xMat,0)
    unReg = bestWeights/varX  #还原后的回归系数
    constant = -1*sum(multiply(meanX,unReg)) + mean(yMat) #常数项
    print("the best model from Ridge Regression is:\n",unReg)
    print("with constant term: ",constant)
    return unReg,constant
    
    
if __name__ == "__main__":
    '''乐高玩具价格预测'''
    #爬取数据
#    setDataCollect()
    #读取数据，这里已将以上方式获取到的数据文本整合成为一个文件即legoAllData.txt
    xmat,ymat = loadDataSet("data/legoAllData.txt")
    #添加对应常数项的特征X0(X0=1)
#    lgX=mat(ones((76,5)))    
#    lgX[:,1:5]=mat(xmat)
#    lgY=mat(ymat).T
    
    #用标准回归方程拟合
#    ws1=standRegres(lgX,mat(ymat)) #求标准回归系数
#    yHat = lgX*ws1 #预测值
#    err1 = rssError(lgY.A,yHat.A)   #计算平方误差
#    cor1 = corrcoef(yHat.T,lgY.T) #计算预测值和真实值得相关性
    
    #用交叉验证测试岭回归    
    ws2,constant = crossValidation(xmat,ymat,10)    
    yHat2 = mat(xmat)*ws2.T + constant
    err2 = rssError(lgY.A,yHat2.A)
    cor2 = corrcoef(yHat2.T,lgY.T)
    
    '''前向逐步线性回归'''    
#    abX,abY=loadDataSet('abalone.txt')
#    stageWise(abX,abY,0.01,200)
    
    '''岭回归'''
#    abX,abY=loadDataSet('abalone.txt')
#    ridgeWeights = ridgeTest(abX,abY)#得到30组回归系数
#    #缩减效果图
#    import matplotlib.pyplot as plt
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    ax.plot(ridgeWeights)
#    plt.show()    
    
    '''线性回归'''
    xArr,yArr=loadDataSet('ex0.txt')
#    ws=standRegres(xArr,yArr)
#    xMat=mat(xArr)
#    yMat=mat(yArr)
#    #预测值
#    yHat=xMat*ws
#    
#    #计算预测值和真实值得相关性
#    corrcoef(yHat.T,yMat)#0.986
#    
#    #绘制数据集散点图和最佳拟合直线图
#    #创建图像并绘出原始的数据
#    import matplotlib.pyplot as plt
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
#    #绘最佳拟合直线，需先要将点按照升序排列
#    xCopy=xMat.copy()
#    xCopy.sort(0)
#    yHat = xCopy*ws
#    ax.plot(xCopy[:,1],yHat)
#    plt.show()
    
    '''局部加权线性回归'''
#    xArr,yArr=loadDataSet('ex0.txt')
#    #拟合
#    yHat=lwlrTest(xArr,xArr,yArr,0.01)
#    #绘图
#    xMat=mat(xArr)
#    yMat=mat(yArr)
#    srtInd = xMat[:,1].argsort(0)
#    xSort=xMat[srtInd][:,0,:]    
#    import matplotlib.pyplot as plt
#    fig=plt.figure()
#    ax=fig.add_subplot(111)
#    ax.plot(xSort[:,1],yHat[srtInd])
#    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0],s=2,c='red')
#    plt.show()
    
    
    
    
