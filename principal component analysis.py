from numpy import *

def loadData(filename, seq="\t"):
    f=open(filename)
    rawData = [i.strip().split(seq) for i in f.readlines()]
    data = [list(map(float, i)) for i in rawData]
    return mat(data)

def pca(dataMat, top=1):
    meanData = dataMat.mean(axis=0)
    moveMean = dataMat - meanData
    covMat = cov(moveMean, rowvar=0)
    eigenValue, eigenVector = linalg.eig(covMat)
    eigenVector = mat(eigenVector)
    topIndex = argsort(eigenValue)
    selectedIndex = topIndex[-top:][::-1]
    transform = eigenVector[:, selectedIndex]
    transformedData = moveMean * transform
    finalData = transformedData * transform.T + meanData
    return transformedData, finalData

def newData():
    filename = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch13\\secom.data"
    data = loadData(filename, seq=" ")
    ncol = data.shape[1]
    for i in range(ncol):
        meanvalue = mean(data[~isnan(data[:,i].A.reshape(-1)),i])
        data[isnan(data[:,i].A.reshape(-1)),i] = meanvalue
    return data
