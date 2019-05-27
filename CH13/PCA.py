import  numpy as np

def loadDataSet(fileName=r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch13\testSet.txt'):
    with open(fileName) as fr:
        stringArr=[line.strip().split() for line in fr.readlines()]
        datArr=[list(map(float,line)) for line in stringArr]
    return np.mat(datArr)

def pca(dataMat,topNfeat=9999999):
    meanVals=np.mean(dataMat,axis=0)
    meanRemoved=dataMat-meanVals
    covMat=np.cov(meanRemoved,rowvar=0) #0-column nonzero-row
    eigVals,eigVects=np.linalg.eig(np.mat(covMat))
    eigValInd=np.argsort(eigVals)
    eigValInd=eigValInd[-1:-(topNfeat+1):-1]
    redEigVects=eigVects[:,eigValInd]
    lowDDataMat=meanRemoved*redEigVects
    reconMat=(lowDDataMat*redEigVects.T)+meanVals
    return  lowDDataMat,reconMat

def replaceNanWithMean():
    datMat=loadDataSet(r'C:\Users\zhang\Desktop\ml\machinelearninginaction\Ch13\secom.data')
    numFeat=np.shape(datMat)[1]
    for i in range(numFeat):
        meanVal=np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0],i])  #
        datMat[np.nonzero(np.isnan(datMat[:,i]))[0],i]=meanVal
    return datMat




'''
import  PCA
dataMat=PCA.loadDataSet()
lowDMat,reconMat=PCA.pca(dataMat,1)
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=90)
ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
plt.show()
datMat=PCA.replaceNanWithMean()
meanVals=np.mean(datMat,axis=0)
meanRemoved=datMat-meanVals
covMat=np.cov(meanRemoved,rowvar=0)
eigVals,eigVects=np.linalg.eig(np.mat(covMat))
print(eigVals)
'''
