# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:40:26 2016
使用python实现的KNN算法进行分类的一个实例，
使用数据集是iris虹膜数据集，保存在iris.txt文件中
@author: Vanguard
"""

import csv     #用于处理csv文件
import random    #用于随机数
import math         
import operator  #
from sklearn import neighbors

#加载数据集
def loadDataset(filename,split,trainingSet=[],testSet = []):
    with open(filename,"rt") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random()<split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[y])

#计算距离
def euclideanDistance(instance1,instance2,length):
    distance = 0
    for x in range(length):
        distance = pow((instance1[x] - instance2[x]),2)
    return math.sqrt(distance)

#返回K个最近邻
def getNeighbors(trainingSet,testInstance,k):
    distances = []
    length = len(testInstance) -1
    #计算每一个测试实例到训练集实例的距离
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    #对所有的距离进行排序
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    #返回k个最近邻
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#对k个近邻进行合并，返回value最大的key
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response]+=1
        else:
            classVotes[response] = 1
    #排序
    sortedVotes = sorted(classVotes.items(),key = operator.itemgetter(1),reverse =True)
    return sortedVotes[0][0]

#计算准确率
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct+=1
    return (correct/float(len(testSet))) * 100.0

def main():
    trainingSet = []  #训练数据集
    testSet = []      #测试数据集
    split = 0.67      #分割的比例
    loadDataset(r"iris.txt", split, trainingSet, testSet) 
    print("Train set :" + repr(len(trainingSet)))
    print("Test set :" + repr(len(testSet)))
    
    predictions = []
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        print(">predicted = " + repr(result) + ",actual = " + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy:" + repr(accuracy) + "%")

if __name__ =="__main__":
    main()



