# -*- coding: utf-8 -*-
'''
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    '''
    在0-m中随机返回一个不等于i的数    
    '''
    j=i 
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    #小于L或大于H的aj将被调整为L或H
    if aj > H: 
        aj = H
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''
    简化版SMO算法
    5个输入参数分别为：数据集、类别标签、常数C、容错率、循环次数    
    '''
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        #用于记录alpha是否已经优化
        alphaPairsChanged = 0
        for i in range(m):
            #fXi为预测类别
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            #Ei预测值与实际的误差         
            Ei = fXi - float(labelMat[i])
            #如果Ei大于容错，且0<alpha<C 进入优化
            #由于后面alpha小于0或大于C时将被调整为0或C，所以一旦在该if语句中它们等于这两个值的话，
            #那么它们就已经在“边界”上了，因而不再能够减小或增大，因此也就不值得再对它们进行优化了。
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                #随机选择第二个alpha
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                #计算L和H。保证alpha在0和C之间
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                #如果L==H，本次循环结束
                if L==H: print("L==H"); continue
                #eta是alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0: print("eta>=0"); continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                #alpha[j]轻微改变，退出for循环
                if (abs(alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); continue
                #对alphas[i]进行修改，修改量与alpha[j]相同，但是方向相反
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                #对alpha[i]和alpha[j]进行优化之后，设置一个常数项b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): iter += 1
        else: iter = 0
        print("iteration number: %d" % iter)
    return b,alphas

def kernelTrans(X, A, kTup): 
    '''
    核函数 
    X(m,n):支持向量集
    A(1,n):待变换的向量
    kTup:含两个参数--①所用核函数的类型 ②速度参数sigma 
    输出K(m,1)
    '''    
    m,n = shape(X)
    K = mat(zeros((m,1)))
    if kTup[0]=='lin': K = X * A.T   #线性核
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T
        K = exp(K/(-1*kTup[1]**2)) #
    else: raise NameError('Error：That Kernel is not recognized！')
    return K

class optStruct:
    '''数据结构保存数据'''
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  #初始化参数
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2))) #第一列为有效标志flag
        self.K = mat(zeros((self.m,self.m))) #for kernel
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
        
def calcEk(oS, k):
    '''计算预测误差'''
#    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)#changed for kernel
    Ek = fXk - float(oS.labelMat[k])
    return Ek    
        
def selectJ(i, oS, Ei):
    '''
    选择第二个alpha，即内循环的alpha值
    依据最大步长(max(abs(Ei - Ej)))选择
    '''
    maxK = -1; maxDeltaE = 0; Ej = 0
    #存储Ei，第一位为有效标记
    oS.eCache[i] = [1,Ei]
    #构建一个非零表，返回非零E值所对应的index    
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   
            if k == i: continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:#第一次循环随机选一个alpha值
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    '''计算误差值，并存入缓存中'''
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    '''
    内循环函数
    此处代码集合和smoSimple()函数一模一样，主要变化有
    1.数据用参数oS传递
    2.用selectJ函数替代selectJrand来选择第二个alpha的值
    3.在alpha值改变时更新Ecache
    
    '''
    Ei = calcEk(oS, i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
        j,Ej = selectJ(i, oS, Ei) #此处和简化版不同
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H: print("L==H"); return 0
#        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j] #changed for kernel
        if eta >= 0: print("eta>=0"); return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
        updateEk(oS, j) #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])
        updateEk(oS, i) #更新误差缓存
#        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
        #for kernel        
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
        else: oS.b = (b1 + b2)/2.0
        #alpha值发生变动返回1，否则返回0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    '''
    主函数--外循环
    '''    
    #用构建的数据结构容纳所有数据    
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0
    #第一个alpha值的选择会在两种方式之间进行交替
    #一种方式是在所有数据集上进行单遍扫描，另一种方式则是在非边界alpha中实现单遍扫描
    #这里非边界指的是那些不等于边界0或C的alpha值
    #同时，这里会跳过那些已知的不会改变的alpha值
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
        alphaPairsChanged = 0
        if entireSet:   
            #在数据集上遍历任意可能的alpha
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        else:#遍历非边界alpha值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
            iter += 1
        if entireSet: entireSet = False #控制循环的flag
        elif (alphaPairsChanged == 0): entireSet = True  
        print("iteration number: %d" % iter)
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    '''
    实际计算出的alpha值为0，而非零alpha所对应的也就是支持向量
    本计算函数遍历所有alpha，但最终起作用的只有支持向量
    '''
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    '''
    利用核函数进行分类的径向基测试函数    
    '''    
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    #训练
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #获取支持向量
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    #分类
    errorCount = 0
    for i in range(m):
        #数据转换
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        #将转换后的数据与前面的alpha及类标签值求积得到预测值        
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    #测试
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))
    
def img2vector(filename):
    """
    将图片数据矩阵转换为向量
    每张图片是32*32像素，也就是一共1024个字节。
    因此转换的时候，每行表示一个样本，每个样本含1024个字节。
    """
    #每个样本数据是1024=32*32个字节
    returnVect = zeros((1,1024))
    fr = open(filename)
    #循环读取32行，32列。
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    '''
    导入数据    
    '''
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        #从文件名中解析出当前图像的标签，也就是数字是几
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        #是数字9类标签设为-1，否则为+1
        if classNumStr == 9: hwLabels.append(-1)
        else: hwLabels.append(1)
        #将图片数据转换为向量
        trainingMat[i,:] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    '''
    基于SVM的手写数字识别
    '''
    dataArr,labelArr = loadImages('digits/trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] 
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors" % shape(sVs)[0])
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount)/m))
    dataArr,labelArr = loadImages('digits/testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f" % (float(errorCount)/m))


'''#######********************************
Non-Kernel VErsions below
'''#######********************************

#class optStructK:
#    def __init__(self,dataMatIn, classLabels, C, toler):  # Initialize the structure with the parameters 
#        self.X = dataMatIn
#        self.labelMat = classLabels
#        self.C = C
#        self.tol = toler
#        self.m = shape(dataMatIn)[0]
#        self.alphas = mat(zeros((self.m,1)))
#        self.b = 0
#        self.eCache = mat(zeros((self.m,2))) #first column is valid flag
#        
#def calcEkK(oS, k):
#    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
#    Ek = fXk - float(oS.labelMat[k])
#    return Ek
#        
#def selectJK(i, oS, Ei):         #this is the second choice -heurstic, and calcs Ej
#    maxK = -1; maxDeltaE = 0; Ej = 0
#    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
#    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
#    if (len(validEcacheList)) > 1:
#        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
#            if k == i: continue #don't calc for i, waste of time
#            Ek = calcEk(oS, k)
#            deltaE = abs(Ei - Ek)
#            if (deltaE > maxDeltaE):
#                maxK = k; maxDeltaE = deltaE; Ej = Ek
#        return maxK, Ej
#    else:   #in this case (first time around) we don't have any valid eCache values
#        j = selectJrand(i, oS.m)
#        Ej = calcEk(oS, j)
#    return j, Ej
#
#def updateEkK(oS, k):#after any alpha has changed update the new value in the cache
#    Ek = calcEk(oS, k)
#    oS.eCache[k] = [1,Ek]
#        
#def innerLK(i, oS):
#    Ei = calcEk(oS, i)
#    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):
#        j,Ej = selectJ(i, oS, Ei) #this has been changed from selectJrand
#        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
#        if (oS.labelMat[i] != oS.labelMat[j]):
#            L = max(0, oS.alphas[j] - oS.alphas[i])
#            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
#        else:
#            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
#            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
#        if L==H: print("L==H"); return 0
#        eta = 2.0 * oS.X[i,:]*oS.X[j,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
#        if eta >= 0: print("eta>=0"); return 0
#        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta
#        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)
#        updateEk(oS, j) #added this for the Ecache
#        if (abs(oS.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
#        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])#update i by the same amount as j
#        updateEk(oS, i) #added this for the Ecache                    #the update is in the oppostie direction
#        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[i,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[i,:]*oS.X[j,:].T
#        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.X[i,:]*oS.X[j,:].T - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.X[j,:]*oS.X[j,:].T
#        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]): oS.b = b1
#        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]): oS.b = b2
#        else: oS.b = (b1 + b2)/2.0
#        return 1
#    else: return 0
#
#def smoPK(dataMatIn, classLabels, C, toler, maxIter):    #full Platt SMO
#    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
#    iter = 0
#    entireSet = True; alphaPairsChanged = 0
#    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
#        alphaPairsChanged = 0
#        if entireSet:   #go over all
#            for i in range(oS.m):        
#                alphaPairsChanged += innerL(i,oS)
#                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
#            iter += 1
#        else:#go over non-bound (railed) alphas
#            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
#            for i in nonBoundIs:
#                alphaPairsChanged += innerL(i,oS)
#                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged))
#            iter += 1
#        if entireSet: entireSet = False #toggle entire set loop
#        elif (alphaPairsChanged == 0): entireSet = True  
#        print("iteration number: %d" % iter)
#    return oS.b,oS.alphas
    
    
if __name__ == "__main__":
#    dataMat,labelMat = loadDataSet('testSet.txt')
#    b,alphas = smoSimple(dataMat,labelMat,0.6,0.001,40)
#    print(b)
#    print(alphas[alphas>0])
#    #支持向量的位置
#    for i in range(100):
#        if alphas[i]>0.0: print(dataMat[i],labelMat[i])
    '''
    完整版
    '''
#    b,alphas = smoP(dataMat,labelMat,0.6,0.001,40)
    
#    testRbf()
    
    testDigits()
    
    
