import numpy as np
import  operator

def creatDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])    //已分类数据坐标,4个点 2个类
    labels=['A','A','B','B']                               //tag for every dataset
    return group,labels

def classify0(inX,dataSet,labels,k):       //inX 分类目标
    //计算inX与每个点的距离
    diffmat=dataSet-inX                                
    sqdiffmat=diffmat**2
    sqdiffmat=np.sum(sqdiffmat,axis=1)
    distance=sqdiffmat**0.5
    
    sorteddistanceindex=distance.argsort()   //根据以上计算的距离排序
    ans={}
    for i in range(k):                  //对比前K个数据，对类别计数
        ans[labels[sorteddistanceindex[i]]] =ans.get(labels[sorteddistanceindex[i]],0)+1  
        # vote=labels[sorteddistanceindex[i]]
        # ans[vote]=ans.get(vote,0)+1
    sortedclasscount=sorted(ans.items(),key=operator.itemgetter(1),reverse=True)  //对ans排序，从大到小
    return sortedclasscount[0][0]  //返回次数最多的类

