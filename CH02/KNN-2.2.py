import numpy as np
import  operator
from pylab import  *
import matplotlib.pyplot as plt

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

#2.2
def file2matrix(filename):    #读取数据，分别存储数据行和对应label
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=np.zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        linesplit=line.split('\t')   #按\t分割，返回结果为列表[]
        returnMat[index,:]=linesplit[0:3]    
        classLabelVector.append(int(linesplit[-1]))  #每行数据最后一列为tag
        index+=1
    return  returnMat,np.array(classLabelVector)

def plot_KNN(datingDataMat,datingLabels,n,m):    #对数据可视化
    fig = plt.figure()
    ax = fig.add_subplot(111)
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    kind = list(set(datingLabels))
    markers = ['o', '*', '+', 'x', 's', 'p', 'h']
    col = ['b', 'r', 'g', 'c', 'y', 'm', 'k']
    label_3 = [r'不喜欢', r'一般', r'特别']
    for i in range(len(kind)):
        xx = datingDataMat[datingLabels == kind[i]]
        yy = datingDataMat[datingLabels == kind[i]]
        plt.scatter(xx[:, n], yy[:, m], marker=markers[i], c=col[i], label=label_3[i])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

#数据归一化
def autoNorm(dataset):
    minval=dataset.min(0)
    maxval=dataset.max(0)
    newval=(dataset-minval)/(maxval-minval)
    return newval

def datingClassTest(filename,test_ratio=0.1,k=3):  #测试集比例为10%
    datingDataMat,datingLabels=file2matrix(filename)
    normMat=autoNorm(datingDataMat)
    m=normMat.shape[0]
    test_num=int(test_ratio*m)
    count=0.0
    for i in range(test_num): 
        classify_ans=classify0(normMat[i,:],normMat[test_num:m,:],
                               datingLabels[test_num:m],k)
        print('the classifier came back with {:2d},real:{:2d}'.format(classify_ans,datingLabels[i]))  #对测试集分类
        if classify_ans!=datingLabels[i]:    #对比分类正确性，记录错误次数
            count+=1.0
    print('error:{:.3f}'.format(count/test_num)) 
