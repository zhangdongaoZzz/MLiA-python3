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

def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()

def ridgeRegress(xMat,yMat,lam=0.2):
    xTx=xMat.T*xMat
    denom=xTx+np.eye(len(xMat))*lam
    if np.linalg.det(denom)==0:
        return 'no'
    ws=denom.I*xMat.T*yMat
    return  ws

def ridgeTest(xArr,yArr):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    yMean=np.mean(yMat,0)
    yMat=yMat-yMean
    xMeans=np.mean(xMat,0)
    xVar=np.var(xMat,0)
    xMat=(xMat-xMeans)/xVar
    numTestPts=30
    wMat=np.zeros((numTestPts,len(xMat[0])))
    for i in range(numTestPts):
        ws=ridgeRegress(xMat,yMat,np.exp(i-10))
        wMat[i,:]=ws.T
    return wMat

def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=np.mat(xArr);yMat=np.mat(yArr).T
    yMat=yMat-np.mean(yMat,0)
    xMat=(xMat-np.mean(xMat,0))/np.var(xMat,0)
    m,n=np.shape(xMat)
    returnMat=np.zeros((numIt,n))
    ws=np.zeros((n,1));wsTest=ws.copy();wsBest=ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError=np.inf
        for j in range(n):
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                yTest=xMat*wsTest
                rssE=rssError(yMat.A,yTest.A)  #to array
                if rssE<lowestError:
                    lowestError=rssE
                    wsBest=wsTest
        ws=wsBest.copy()
        returnMat[i,:]=ws.T
    return returnMat

#此后直接copy，因为得不到google的数据。 有兴趣的自行改动。
from time import sleep
import json
import urllib2


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (
    myAPIstr, setNum)
    pg = urllib2.urlopen(searchURL)
    retDict = json.loads(pg.read())
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]
            if currItem['product']['condition'] == 'new':
                newFlag = 1
            else:
                newFlag = 0
            listOfInv = currItem['product']['inventories']
            for item in listOfInv:
                sellingPrice = item['price']
                if sellingPrice > origPrc * 0.5:
                    print
                    "%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice)
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        except:
            print
        'problem with item %d' % i


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))  # create error mat 30columns numVal rows
    for i in range(numVal):
        trainX = [];
        trainY = []
        testX = [];
        testY = []
        random.shuffle(indexList)
        for j in range(m):  # create training set based on first 90% of values in indexList
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX, trainY)  # get 30 weight vectors from ridge
        for k in range(30):  # loop over all of the ridge estimates
            matTestX = mat(testX);
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain  # regularize test with training params
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)  # test ridge results and store
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
            # print errorMat[i,k]
    meanErrors = mean(errorMat, 0)  # calc avg performance of the different ridge weight vectors
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    # can unregularize to get model
    # when we regularized we wrote Xreg = (x-meanX)/var(x)
    # we can now write in terms of x not Xreg:  x*w/var(x) - meanX/var(x) +meanY
    xMat = mat(xArr);
    yMat = mat(yArr).T
    meanX = mean(xMat, 0);
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    print
    "the best model from Ridge Regression is:\n", unReg
    print
    "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)
