import matplotlib.pyplot as plt

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va="center",ha="center",bbox=nodeType
                            ,arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getNumLeafs(myTree):
    numLeafs=0
    firstStr=myTree.key()[0]
    secondDict=myTree[firstStr]
    for key in secondDict.key():
        if type(secondDict[key]).__name__=='dict':
            numLeafs+=getNumLeafs(secondDict[key])
        else:
            numLeafs+=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth=0
    firstStr=myTree.key()[0]
    secondDict=myTree[firstStr]
    for key in secondDict:
        if type(secondDict[key]).__name__=='dict':
            thisdepth=1+getTreeDepth(secondDict[key])
        else:
            thisdepth=1
        if thisdepth>maxDepth:maxDepth=thisdepth
    return  maxDepth
