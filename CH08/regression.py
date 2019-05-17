import  numpy as np


def loadDataSet(filename=r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch08\ex0.txt'):
    numFeat=len(open(filename).readline().split())
    dataMat=[];labelMat=[]
    with open(filename) as  fr:
        for line in fr.readlines():
            currLine=line.strip().split()
            dataMat.append([float(currLine[i]) for i in range(numFeat-1)])
            labelMat.append(float(currLine[-1]))
    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    xTx=xMat.T*xMat
    if np.linalg.det(xTx)==0:    #线性代数库linalg det 计算行列式
        print('can t do this')
        return -1
    x1=xTx.I
    ws=xTx.I*xMat.T*yMat
    return ws

def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    m=len(xMat)
    weights=np.eye((m))
    for i in range(m):
        diffMat=testPoint-xMat[i,:]
        weights[i][i]=np.exp((diffMat*diffMat.T)/(-2.0*k**2))
    xTx=xMat.T*weights*xMat
    if np.linalg.det(xTx)==0:
        print('no')
        return
    ws=xTx.I*xMat.T*weights*yMat
    return testPoint*ws

def lwlrTest(testArr,xArr,yArr,k=1.0):
    m=len(testArr)
    yHat=[float(lwlr(testArr[i],xArr,yArr,k)) for i in range(m)]
    return np.array(yHat)

