__author__ = 'Administrator'
#coding=utf-8
'''
梯度上升算法
'''
import sys
from numpy import *

sys.path.append('D:/utopiar/PycharmProjects/logRegres')

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('D:/utopiar/PycharmProjects/logRegres/data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn, classLabels):
    #将矩阵转换为numpy类型
    dataMatrix = mat(dataMatIn)  #dataMatIn、dataMatrix为n*3的矩阵
    labelMatrix = mat(classLabels).transpose() #classLabels为n维行向量，labelMatrix为n为列向量
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1)) #weights为n维列向量
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMatrix - h)
        '''
        dataMatrix.transpose() * error
                3*n维             n*1维
        '''
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights

def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        print array(dataMatrix[i])
        print weights
        print type(weights)
        '''
        dataMatrix[i]为List类型，不能直接进行乘法运算，假如a是一个List类型，那么2*a的结果是将a复制一份
        '''
        weights = weights + alpha * error * array(dataMatrix[i])
    return weights

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in dataIndex:
            alpha = 4/(1.0+j+i)+0.01
            randIndex = int (random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * array(dataMatrix[randIndex])
            del(dataIndex[randIndex])
    return weights