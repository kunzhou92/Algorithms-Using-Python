import numpy
import matplotlib.pyplot as plt
def create_data():
    f = open("E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch05\\testSet.txt")
    data_list = []
    label_list = []
    for row in f.readlines():
        temp_data = row.strip().split()
        data_list.append([1, float(temp_data[0]), float(temp_data[1])])
        label_list.append(int(temp_data[2]))
    return data_list, label_list
                         
def sigmoid(value):
    return 1 / ( 1 + numpy.exp(-value))
        
def gradient_method(data_list, label_list):
    data_mat = numpy.mat(data_list)
    label_mat = numpy.mat(label_list).transpose()
    alpha = 0.001
    max_circle = 500
    nrow, ncol = data_mat.shape
    weight = numpy.mat(numpy.ones((ncol, 1)))
    for i in range(max_circle):
        sigmoid_value = sigmoid(data_mat*weight)
        error = label_mat - sigmoid_value
        weight = weight + data_mat.transpose()*error*alpha
    return weight
    
def plot_fig(wei, data_list, label_list):
    weight = numpy.array(wei)
    fig = plt.figure()
    sub_fig = fig.add_subplot(111)
    x1 = []; y1 = []; x2 = []; y2 = []
    n = len(data_list)
    for i in range(n):
        if label_list[i] == 1:
            x1.append(data_list[i][1]); y1.append(data_list[i][2])
        else:
            x2.append(data_list[i][1]); y2.append(data_list[i][2])
    sub_fig.scatter(x1, y1, s = 30, c="red", marker = "s")
    sub_fig.scatter(x2, y2, s = 30, c="blue")
    x = numpy.arange(-3, 3, 0.1)
    y = -(weight[0] + weight[1]*x) / weight[2]
    sub_fig.plot(x, y)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def stochastic_gradient(data_list, label_list):
    data_arr = numpy.array(data_list)
    label_arr = numpy.array(label_list)
    nrow, ncol = data_arr.shape
    weight = numpy.ones(ncol)
    alpha = 0.01
    for i in range(nrow):
        sigmoid_value = sigmoid(sum(data_arr[i] * weight))
        error = label_arr[i] - sigmoid_value
        weight = weight + error * alpha * data_arr[i]
    return weight
        
def stochatic_gradient_adv(data_list, label_list, numIter=150):
    data_arr = numpy.array(data_list)
    label_arr = numpy.array(label_list)
    nrow, ncol = data_arr.shape
    weight = numpy.ones(ncol)
    for i in range(numIter):
        data_index = list(range(nrow))
        for j in range(nrow):
            alpha = 4 / (1.0+j+i) + 0.01
            data_index_each = int(numpy.random.uniform(0, len(data_index)))
            sigmoid_value = sigmoid(sum(data_arr[data_index_each] * weight))
            error = label_arr[data_index_each] - sigmoid_value
            weight = weight + error * alpha * data_arr[data_index_each]
            del(data_index[data_index_each])
    return weight

def classify_vector(weight, test_vec):
    result = 1 / (1 + numpy.exp(-sum(numpy.array(test_vec) * numpy.array(weight))))
    if result > 0.5:
        return 1
    else:
        return 0

def colic_test():
    f_test = open("E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch05\\horseColicTest.txt")
    f_training = open("E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch05\\horseColicTraining.txt")
    training_data_list = []
    training_label_list = []
    for i in f_training.readlines():
        temp = i.strip().split("\t")
        temp_list = []
        for j in range(21):
            temp_list.append(float(temp[j]))
        training_data_list.append(temp_list)
        training_label_list.append(float(temp[21]))
    weight = stochatic_gradient_adv(training_data_list, training_label_list)
    error = 0
    count = 0
    for i in f_test.readlines():
        count += 1
        temp = i.strip().split("\t")
        temp_list = []
        for j in range(21):
            temp_list.append(float(temp[j]))
        outcome = classify_vector(weight, temp_list)
        if outcome != int(temp[21]):
            error += 1
    error_rate = error / count
    print("error rate is %f" % error_rate)
    return error_rate

def multitest(num = 10):
    count = 0
    error = 0
    for i in range(num):
        error += colic_test()
        count += 1
    print("total error rate  is %f" % (error/float(count)))





    
