import numpy as  np
import matplotlib.pyplot as plt


def loadDataSet(path=r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch05\testSet.txt'):
    with open(path) as fr:
        dataMat = [];
        lableMat = []
        for line in fr.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            lableMat.append(int(lineArr[-1]))
        return dataMat, lableMat
    return -1


def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).reshape(-1, 1)
    m, n = np.shape(dataMatrix)
    weight = np.ones((n, 1))
    alpha = 0.001
    for i in range(500):
        h = sigmoid(dataMatrix * weight)
        error = labelMat - h
        weight = weight + alpha * dataMatrix.T * error
    return weight


def plotBestFit(wei):
    weights=wei
    if type(wei) == 'numpy.matrixlib.defmatrix.matrix':
        weights = weights.getA()  # array
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]);
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]);
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

import  random
def stocGradAscent1(dataMatrix,classLabels,numIter=200):
    m,n=np.shape(dataMatrix)
    weights=np.ones(n)
    for i in range(numIter):
        for j in range(m):
            alpha=4.0/(5+i+j)+0.005
            Index=random.randint(0,m-1)     #这里与书上做法不同，不删除已选的随机数，效果差不多
            h=sigmoid(sum(dataMatrix[Index]*weights))
            error=classLabels[Index]-h
            weights=weights+alpha*error*dataMatrix[Index]
    return weights

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:return 1.0
    else:return 0.0

def colicTest(path=r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch05\{}.txt'):
    frtrain=open(path.format('horseColicTraining'))
    frtest=open(path.format('horseColicTest'))
    trainingSet=[];trainLabels=[]
    for line in frtrain.readlines():
        currline=line.strip().split('\t')
        trainingSet.append([float(currline[i]) for i in range(21)])
        trainLabels.append(float(currline[21]))
    trainWeights=stocGradAscent1(np.array(trainingSet),trainLabels,1500)
    errorCount=0;numTestVec=0.0
    for line in frtest.readlines():
        numTestVec+=1.0
        currline=line.strip().split('\t')
        lineArr=[float(currline[i]) for i in range(21)]
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currline[21]):
            errorCount+=1
    print('error:%f'%(errorCount/numTestVec))
    return errorCount/numTestVec

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print ("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))
