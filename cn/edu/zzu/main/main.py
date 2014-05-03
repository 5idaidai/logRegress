__author__ = 'Administrator'
#coding=utf-8

'''
主函数入口
'''
from cn.edu.zzu.modules import grad
from cn.edu.zzu.modules import plotBestFit


dataArr,labelMat = grad.loadDataSet()
weights = grad.gradAscent(dataArr,labelMat)
weight1s = grad.stocGradAscent0(dataArr,labelMat)
weight2s = grad.stocGradAscent1(dataArr,labelMat)
plotBestFit.plotBestFit(weights.getA())
plotBestFit.plotBestFit(weight1s)
plotBestFit.plotBestFit(weight2s)
