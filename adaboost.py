from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def build_stump(data_arr, label_arr, weight, stepnum):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).T
    nrow, ncol = data_mat.shape
    min_error = inf
    best_class_est = mat(zeros((nrow, 1)))
    best_stump = {}    
    for i in range(ncol):
        fea_min = data_mat[:,i].min()
        fea_max = data_mat[:,i].max()
        step = (fea_max - fea_min) / stepnum
        for j in range(-1,(stepnum+1)):
            threshold = fea_min + float(j) * step
            for k in ["lt", "gt"]:
                expected_return = stumpClassify(data_mat, i, threshold, k)
                error_mat = mat(ones((nrow,1)))
                error_mat[expected_return == label_mat] = 0
                weight_error = mat(weight).T * error_mat
                #print("split: dim %d, thresh %.2f, ineqal: %s, the weighted error is %.3f" %(i, threshold, k, weight_error))
                if weight_error < min_error:
                    min_error = weight_error
                    best_class_est = expected_return
                    best_stump["dim"] = i
                    best_stump["thresh"] = threshold
                    best_stump["ineq"] = k
    return best_stump, min_error, best_class_est

def ada_boost(data_arr, label_arr, stepnum, max_interation=40):
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).T
    nrow, ncol = data_mat.shape
    record_stump = []
    weight = mat(ones((nrow, 1)) / nrow)
    agg_exp = zeros((nrow,1))
    for i in range(max_interation):
        #print("weight: ", weight.T)
        each_stump, each_error, each_est = build_stump(data_mat, label_arr, weight, stepnum)
        alpha = log((1.0 - each_error) / max(each_error,1e-16)) / 2
        expon = -multiply(label_mat, each_est) * alpha
        weight = multiply(weight, exp(expon))
        weight = weight / sum(weight)
        agg_exp += multiply(alpha, each_est)
        each_stump["alpha"] = alpha
        #print("each est: ", each_est.T)
        record_stump.append(each_stump)
        #print("agg_exp: ", agg_exp.T)
        agg_error = multiply(sign(agg_exp) != label_mat, ones((nrow, 1)) )
        error_rate = sum(agg_error) / len(agg_error)
        #print("total error: ", error_rate)
        if error_rate == 0.0:
            break
    return record_stump

def ada_classify(data, classifier):
    data_mat = mat(data)
    nrow, ncol = data_mat.shape
    agg_exp = zeros((nrow, 1))
    for each_class in classifier:
        each_exp = stumpClassify(data_mat,each_class["dim"],each_class["thresh"],each_class["ineq"])
        agg_exp += multiply(each_class["alpha"], each_exp)
        #print(agg_exp)
    return sign(agg_exp), agg_exp

def load_file(filename):
    feature_num = len( open(filename).readline().strip().split("\t") )
    data = []
    label = []
    for row in open(filename).readlines():
        temp = []
        individual_data = row.strip().split("\t")
        for j in range(feature_num-1):
            temp.append(float(individual_data[j]))
        data.append(temp)
        label.append(float(individual_data[-1]))
    return mat(data), label


def plot_roc(pred_strength, class_label):
    import matplotlib.pyplot as plt
    pred_strength = pred_strength.reshape(1, len(pred_strength))
    rank_value = pred_strength[0].argsort()
    point = (1, 1)
    num_pos_class = sum(array(class_label) == 1)
    y_step = 1 / num_pos_class
    x_step = 1 / (len(class_label) - num_pos_class)
    auc = 0
    fig = plt.figure()
    fig.clf()
    subfig = fig.add_subplot(111)
    for i in rank_value.tolist():
        if class_label[i] == 1:
            x = 0; y = y_step
        else:
            x = x_step; y = 0
            auc += x_step * point[1]
        subfig.plot([point[0], point[0]-x], [point[1], point[1]-y], c="b")
        point = (point[0]-x, point[1]-y)
    subfig.plot([0,1], [0,1], "b--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("title")
    subfig.axis([0, 1, 0, 1])
    plt.show()
    print("AUC: ", auc)


















