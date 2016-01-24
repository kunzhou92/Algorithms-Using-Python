from numpy import *

def load_data(filename):     #output array
    f = open(filename)
    data=[]
    for i in f.readlines():
        temp = i.strip().split("\t")
        data.append(list(map(float,temp)))
    return array(data)

def euclidean_distance(vec1, vec2):  #output number, input array, mat
    return sqrt(sum(power(vec1-vec2, 2)))

def random_centroids(data, k): #output mat
    data = mat(data)
    nrow, ncol = data.shape
    result = mat(zeros((k, ncol)))
    max_vec = data.max(axis=0)
    min_vec = data.min(axis=0)
    result = multiply(min_vec,ones((k,1))) + \
                  multiply((max_vec-min_vec), random.rand(k,1))
    return result

def k_means(data, k, dist=euclidean_distance, \
            centroids_selection=random_centroids):
    nrow, ncol = shape(data)
    data = mat(data)
    centroids = centroids_selection(data, k)
    data_record = mat(zeros((nrow,2)))
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(nrow):
            min_dis = inf; min_cluster = -1
            for j in range(k):
                temp_dis = euclidean_distance(centroids[j,:], data[i,:])
                if temp_dis < min_dis:
                    min_dis = temp_dis
                    min_cluster = j
            if data_record[i, 0] != min_cluster:
                cluster_changed = True
            data_record[i, 0] = min_cluster
            data_record[i, 1] = min_dis ** 2
        # print(centroids)
        for j in range(k):
            data_select = data[data_record.A[:,0] == j]
            centroids[j,:] = data_select.mean(axis=0)
    return centroids, data_record

def bisect_kmeans(data, k, dist=euclidean_distance):
    nrow, ncol = shape(data)
    data = mat(data)
    data_record = mat(zeros((nrow, 2)))
    centroid0 = mean(data,axis=0)
    centroids = [centroid0.tolist()[0]]
    for i in range(nrow):
        data_record[i,1] = dist(centroid0, data[i,:]) ** 2
    for i in range(k-1):
        best_see = inf
        for j in range(len(centroids)):
            selected_data = data[data_record.A[:,0] == j]
            sub_centroids, sub_data_record = k_means(selected_data, 2, dist)
            sub_split_sse = sum(sub_data_record[:,1])
            not_selected_data_record = data_record[data_record.A[:,0]!= j]
            try:
                sub_not_split_sse = sum(not_selected_data_record[:,1])
            except Exception:
                sub_not_split_sse = 0
            sub_sse = sub_not_split_sse + sub_split_sse
            print("ssesplit is %f, and notsplit is %f" %(sub_split_sse,\
                                                         sub_not_split_sse))
            if sub_sse < best_see:
                best_see = sub_sse
                best_centroids = sub_centroids
                best_data_record = sub_data_record
                best_index = j
        best_data_record[best_data_record.A[:,0] == 1, 0] = i+1
        best_data_record[best_data_record.A[:,0] == 0, 0] = best_index
        data_record[data_record.A[:,0] == best_index] =  best_data_record
        print("BestCentToSplit: ", best_index)
        print("the len of bestClussAss: ", len(best_data_record))
        centroids[i] = best_centroids[0,:].tolist()[0]
        centroids.append(best_centroids[1,:].tolist()[0])
    return centroids, data_record
        
def distSLC(vecA, vecB):#Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * \
                      cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0 #pi is imported with numpy

import matplotlib
import matplotlib.pyplot as plt
def cluster_club(num=5):
    data =[]
    filename = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch10\\places.txt" 
    for row in open(filename).readlines():
        temp = row.strip().split("\t")
        data.append([float(temp[4]), float(temp[3])])
    data = mat(data)
    centroids, data_record = bisect_kmeans(data, num, distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    markers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label="ax0", **axprops)
    graph = plt.imread("E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch10\\Portland.png")
    ax0.imshow(graph)
    ax1 = fig.add_axes(rect, label="ax1", frameon=False)
    for i in range(len(centroids)):
        data_selected = data[data_record.A[:,0]==i]
        ax1.scatter(data_selected.A[:,0], data_selected.A[:,1], \
                    marker=markers[i%len(markers)], s=90)
    centroids = mat(centroids)
    ax1.scatter(centroids.A[:,0], centroids.A[:,1], marker="+", s=300)
    plt.show()
                    
        


























    
