import numpy as np
import operator

def systemShannonEnt(dataset):
    numEntries = len(dataset)
    labelCounts = {}
    for featVec in dataset:
        currentlabel = featVec[-1]
        if currentlabel not in labelCounts.keys():
            labelCounts[currentlabel] = 0
        labelCounts[currentlabel] += 1
    ShannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key] / numEntries)
        ShannonEnt -= prob * np.log2(prob)  #np.log的参数已经改变，用log2（x），或者np.math.log(x,2)
    return ShannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def splitDataSet(dataset,axis,value):
    retDataSet=[]
    for featVec in dataset:
        if featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return  retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=systemShannonEnt(dataSet)
    bestInfoGain=0.0 ;bestFeature=-1
    for i in range(numFeatures):
        featList=[ex[i] for ex in  dataSet]
        uniqueVals=set(featList)
        newEnt=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEnt+=prob*systemShannonEnt(subDataSet)
        InfoGain=baseEntropy-newEnt
        if InfoGain>bestInfoGain:
            bestInfoGain=InfoGain
            bestFeature=i
    return bestFeature

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys() :
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[ex[-1] for ex in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1:
        return majorityCnt(classList)
    bestFeat=chooseBestFeatureToSplit(dataSet)
    bestFeatLabel=labels[bestFeat]  #mappnig
    myTree={bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues=[ex[bestFeat] for ex in dataSet]
    uniquevals=set(featValues)
    for value in uniquevals:
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),labels)
    return myTree
