import KNN
import matplotlib.pyplot as plt
import  numpy as  np
from pylab import  *



group,labels=KNN.creatDataSet()

datingDataMat,datingLabels=KNN.file2matrix(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch02\datingTestSet2.txt')

KNN.plot_KNN(datingDataMat,datingLabels,0,2)

print(KNN.autoNorm(datingDataMat))

KNN.datingClassTest(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch02\datingTestSet2.txt')
