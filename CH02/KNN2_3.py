import numpy as np
import  operator
from pylab import  *
import matplotlib.pyplot as plt
from  functools import  reduce
import  os
import KNN

def img2vector(filename):
    vector=np.zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            vector[0,32*i+j]=int(line[j])
    return vector

def handwritingClassTest(trainpath=r'yourpath',testpath=r'yourpath'): #将yourpath 改为自己存放数据的路径
    hwLabels=[]
    trainingFileList=os.listdir(trainpath)
    m=len(trainingFileList)
    trainMat=np.zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        label=int(fileNameStr.split('_')[0])
        hwLabels.append(label)
        trainMat[i,:]=img2vector('{}/{:s}'.format(trainpath,fileNameStr))
    testFileList=os.listdir(testpath)
    error=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumStr=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector('{}/{:s}'.format(testpath,fileNameStr))
        classifierResult=KNN.classify0(vectorUnderTest,trainMat,hwLabels,3)
        print('the pre:%d,  the real:%d'%(classifierResult,classNumStr))
        if classifierResult!=classNumStr:
            error+=1
    print('error rate:%f'%(error/mTest))

