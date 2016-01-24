from numpy import *
import operator
import os
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals


def Truetest():
    percentTats = float(input("percentTats"))
    ffMiles = float(input("ffMiles"))
    iceCream = float(input("iceCream"))
    sample = array([percentTats, ffMiles, iceCream])
    data, label = file2matrix("E:\Anaconda3\SourceCode\machinelearninginaction\Ch02\datingTestSet2.txt")
    norm, rangee, minn  = autoNorm(data)
    norm_sample = (sample - minn) / rangee
    result = classify0(norm_sample, norm, label, 3)
    print("result is %s" %(result))
    return norm_sample

def transform_to_vector(filename):
    f = open(filename)
    vector = zeros(1024)
    for i in range(32):
        f_read_line = f.readline()
        for j in range(32):
            vector[i*32+j] = int(f_read_line[j])
    return vector

def handwriting ():
    training_path = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch02\\trainingDigits"
    training_name = os.listdir(training_path)
    data_row_len = len(training_name)
    labels = []
    data = zeros((data_row_len, 1024))
    for i, file in enumerate(training_name):
        title = file.split(".")[0]
        data_label = title.split("_")[0]
        labels.append(data_label)
        file_path = training_path + "\\"+ file
        data[i,:] = transform_to_vector(file_path)
    test_path = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch02\\testDigits"
    test_name = os.listdir(test_path)
    count = 0
    for i, file in enumerate(test_name):
        title = file.split(".")[0]
        data_label = title.split("_")[0]
        file_path = training_path + "\\"+ file
        sample = transform_to_vector(file_path)
        prediction = classify0(sample, data, labels, 3)
        if prediction != data_label:
            count += 1
            print(file)
            print("prediction is %s and real value is %s" %(prediction, data_label))
    rate = count / len(test_name)        
    print("total count is %d and error rate is %f" % (count, rate))
        



