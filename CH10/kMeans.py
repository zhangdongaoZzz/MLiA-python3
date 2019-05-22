import numpy as  np


def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curline = line.strip().split()
            fltline = list(map(float, curline))
            dataMat.append(fltline)
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.power(vecA - vecB, 2).sum())  # not sum(mat) but mat.sum()


def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  ####矩阵怎么用下标取都依旧是矩阵 len(dataset[0])并不能知道列数
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minj = min(dataSet[:, j])
        rangej = float(max(dataSet[:, j]) - minj)
        centroids[:, j] = minj + rangej * np.random.rand(k, 1)
    return centroids


def KMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = randCent(dataSet, k)
    clusterchanged = True
    while clusterchanged:
        clusterchanged = False
        for i in range(m):
            cluster_i = clusterAssment[i, 0];
            dismax = np.inf
            for j in range(k):
                curdis = distEclud(centroids[j, :], dataSet[i, :])
                if curdis < dismax:
                    dismax = curdis
                    clusterAssment[i, :] = j, dismax
            if cluster_i != clusterAssment[i, 0]: clusterchanged = True
        print(centroids)
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]
    cenList = [centroid0]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :]) ** 2
    while len(cenList) < k:
        lowestSSE = np.inf
        for i in range(len(cenList)):
            ptscurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            centroidMat, splitClustAss = KMeans(ptscurrCluster, 2)
            ssesplit = splitClustAss[:, 1].sum()
            ssenotsplit = clusterAssment[np.nonzero(clusterAssment[:, 0].A != i), 1].sum()
            print(ssesplit, ssenotsplit)
            if ssesplit + ssenotsplit < lowestSSE:
                lowestSSE = ssenotsplit + ssesplit
                bestnewCent = centroidMat
                bestClustAss = splitClustAss.copy()
                bestCentToSplit = i
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(cenList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestcenttosplit: ', bestCentToSplit)
        print('len bestclustass: ', len(bestClustAss))
        cenList[bestCentToSplit] = bestnewCent[0, :]
        cenList.append(bestnewCent[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],
        :] = bestClustAss  # reassign new clusters, and SSE
    return np.mat(cenList), clusterAssment

#####未改### not alter ####
import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print yahooApi
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print "%s\t%f\t%f" % (lineArr[0], lat, lng)
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else: print "error fetching"
        sleep(1)
    fw.close()



######################
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = np.sin(vecA[0,1]*np.pi/180) * np.sin(vecB[0,1]*np.pi/180)
    b = np.cos(vecA[0,1]*np.pi/180) * np.cos(vecB[0,1]*np.pi/180) * \
                      np.cos(np.pi * (vecB[0,0]-vecA[0,0]) /180)
    return np.arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scatterMarkers=['s', 'o', '^', '8', 'p', \
                    'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

