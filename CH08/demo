import  regression
xArr,yArr=regression.loadDataSet()
# print(xArr[0:2])
w=regression.standRegres(xArr,yArr)
# print(w)

print(yArr[0],regression.lwlr(xArr[0],xArr,yArr,0.001))

yHat=regression.lwlrTest(xArr,xArr,yArr,0.003)
print(yHat  )
xMat=np.mat(xArr)
srtInd=xMat[:,1].argsort(0)  #?
xSort=xMat[srtInd][:,0,:]    #?

fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(xSort[:,1],yHat[srtInd])
ax.scatter(xMat[:,1].flatten().A[0],mat(yArr).T.flatten().A[0],s=2,c='red')
plt.show()
