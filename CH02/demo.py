import KNN
import matplotlib.pyplot as plt
import  numpy as  np
from pylab import  *


#2.2
group,labels=KNN.creatDataSet()

datingDataMat,datingLabels=KNN.file2matrix(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch02\datingTestSet2.txt')

KNN.plot_KNN(datingDataMat,datingLabels,0,2)

print(KNN.autoNorm(datingDataMat))

KNN.datingClassTest(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch02\datingTestSet2.txt')

#2.3
c=KNN2.img2vector(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch02\trainingDigits\0_0.txt')
KNN2.handwritingClassTest()
