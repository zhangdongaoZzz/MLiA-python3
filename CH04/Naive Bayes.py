import  numpy as np



def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def  createVocabList(dataset):
    vocabSet=set([ vocab for x in dataset for vocab in x ])  #列表推导
    return  list(vocabSet)

def setOfwords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for vocab in  inputSet:
        if vocab in vocabList:
            returnVec[vocabList.index(vocab)]=1
        else:
            print('not exist')
    return returnVec



def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones()
    p0Denom = 2.0; p1Denom = 2.0                        #change to 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)          #change to log()
    p0Vect = np.log(p0Num/p0Denom)          #change to log()
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0,p1,pA):
    p1=sum(vec2Classify*p1)+np.log(pA)
    p0=sum(vec2Classify*p0)+np.log(pA)
    if p0>p1:return 0
    else: return 1


def testingNB():
    listOPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]
    for postinDoc in listOPosts:
        trainMat.append(setOfwords2Vec(myVocabList,postinDoc))
    p0,p1,pA=trainNB0(np.array(trainMat),np.array(listClasses))
    testEntry=['love','my','dalmation']
    thisDoc=np.array(setOfwords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0,p1,pA))


def bagOfwords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
    return  returnVec

