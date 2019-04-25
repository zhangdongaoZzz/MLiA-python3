import numpy as np
import  operator

def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    diffmat=dataSet-inX
    sqdiffmat=diffmat**2
    sqdiffmat=np.sum(sqdiffmat,axis=1)
    distance=sqdiffmat**0.5
    sorteddistanceindex=distance.argsort()
    ans={}
    for i in range(k):
        ans[labels[sorteddistanceindex[i]]] =ans.get(labels[sorteddistanceindex[i]],0)+1
        # vote=labels[sorteddistanceindex[i]]
        # ans[vote]=ans.get(vote,0)+1
    sortedclasscount=sorted(ans.items(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

