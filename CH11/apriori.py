import  numpy as np

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1=[]
    for transaction in dataSet:
        for item in transaction:
            if [item] not in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset,C1))  #python2 map 返回迭代器

def scanD(Data,Ck,minSupport):  #
    ssCnt={}
    for tid in Data:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.__contains__(can):  #python2:ssCnt.has_key(can)
                    ssCnt[can]=1
                else:
                    ssCnt[can]+=1
    numItems=float(len(Data))  #python2 no list()
    retList=[]
    supportData={}
    for key in ssCnt.keys():
        support=ssCnt[key]/numItems
        supportData[key]=support
        if support>=minSupport:
            retList.insert(0,key)
    return retList,supportData

def aprioriGen(Lk,k):
    retList=[]
    lenLk=len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]
            L1.sort();L2.sort()
            if L1==L2:
                retList.append(Lk[i]|Lk[j])
    return retList

def apriori(dataSet,minSupport=0.5):
    C1=createC1(dataSet)
    D=list(map(set,dataSet))
    L1,supportData=scanD(D,C1,minSupport)
    L=[L1]
    k=2
    while (len(L[k-2])>0):
        Ck=aprioriGen(L[k-2],k)
        Lk,supK=scanD(D,Ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k+=1
    return L,supportData
