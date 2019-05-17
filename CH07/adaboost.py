import numpy as np
import operator


def loadSimpData():
    datMat = np.matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray=np.ones((len(dataMatrix),1))
    if threshIneq=='lt':
        retArray[dataMatrix[:,dimen]<=threshVal]=-1.0
    else:
        retArray[dataMatrix[:,dimen]>threshVal]=-1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix=np.mat(dataArr);labelMat=np.mat(classLabels).T
    m,n=np.shape(dataMatrix)
    numSteps=10.0;bestStump={};bestClasEst=np.mat(np.zeros((m,1)))
    minError=np.inf
    for i in range(n):
        rangemin=dataMatrix[:,i].min();rangemax=dataMatrix[:,i].max()
        stepSize=(rangemax-rangemin)/numSteps
        for j in range(0,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal=(rangemin+float(j)*stepSize)
                predictedVals=stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr=np.mat(np.ones((m,1)))
                errArr[predictedVals==labelMat]=0
                weightedError=D.T*errArr
                # print('split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f'% (i, threshVal, inequal, weightedError))
                if weightedError<minError:
                    minError=weightedError
                    bestClasEst=predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClass=[]
    m=len(dataArr)
    D=np.mat(np.ones((m,1))/m)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst=buildStump(dataArr,classLabels,D)
        # print(D.T)
        alpha=float(0.5*np.log((1-error)/max(error,1e-16)))
        bestStump['alpha']=alpha
        weakClass.append(bestStump)
        # print('classEst',classEst.T)
        expon=np.multiply(-1*alpha*np.mat(classLabels).T,classEst)
        D=np.multiply(D,np.exp(expon))
        D=D/sum(D)
        aggClassEst+=alpha*classEst
        # print('aggClassEst',aggClassEst)
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        aggErrors=np.multiply(np.sign(aggClassEst)!=np.mat(classLabels).T,np.ones((m,1)))
        errorRate=sum(aggErrors)/m
        print('total error:',errorRate)
        if errorRate==0.0:
            break
    return  weakClass


def adaClassify(datToClass,classifierArr):
    dataMatrix=np.mat(datToClass)
    m=len(dataMatrix)
    aggClassEst=np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst=stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst+=classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return np.sign(aggClassEst)

def loadDataSet(filename=r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch07\horseColicTraining2.txt'):
    numFeat=len(open(filename).readline().split())
    dataMat=[];labelMat=[]
    with open(filename) as  fr:
        for line in fr.readlines():
            currLine=line.strip().split()
            dataMat.append([float(currLine[i]) for i in range(numFeat-1)])
            labelMat.append(float(currLine[-1]))
    return dataMat,labelMat
