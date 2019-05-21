import numpy as np


def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        for Line in fr.readlines():
            currLine = Line.strip().split()
            fltLine = list(map(float,
                               currLine))  # python3 中map返回的是迭代器，而python2返回列表。https://www.runoob.com/python/python-func-map.html
            dataMat.append(fltLine)
        # dataMat=[list(map(float,Line.strip().split()) for Line in fr.readlines())]
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    # x1=dataSet[:,feature]>value
    # x2=np.nonzero(dataSet[:,feature]>value)
    # x3=dataSet[np.nonzero(dataSet[:,feature]>value)[0],:]
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0],
           :]  # [0]  #https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]  # [0]
    return mat0, mat1


def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    return np.var(dataSet[:, -1]) * len(dataSet)


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=[1, 4]):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None: return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=[1, 4]):
    tolS = ops[0];
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errType(dataSet)
    bestS = np.inf;
    bestIndex = 0;
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if len(mat0) < tolN or len(mat1) < tolN: continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = splitVal
            if (S - bestS) < tolS:
                return None, leafType(dataSet)
            mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
            if len(mat0) < tolN or len(mat1) < tolN:
                return None, leafType(dataSet)
    return bestIndex, bestValue


def isTree(obj):
    return type(obj).__name__ == 'dict'


def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2


def prune(tree, testData):
    if len(testData) == 0: return getMean(tree)
    if (isTree(tree['right'])) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['right']) and not isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree

def linearSolve(dataSet):
    m,n=np.shape(dataSet)
    X=np.mat(np.ones((m,n)));Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if np.linalg.det(xTx)==0:
        return 'no'
    ws=xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataSet):
    ws,X,Y=linearSolve(dataSet)
    return ws
def modelErr(dataSet):
    ws,X,Y=linearSolve(dataSet)
    yHat=X*ws
    return sum(np.power(Y-yHat,2))


def regTreeEval(model, inDat):
    return float(model)


def modelTreeEval(model, inDat):
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)
    yHat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat
