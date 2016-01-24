from numpy import *

def load_data(filename):    # output array
    f = open(filename)
    data = []
    for row in f.readlines():
        temp = row.strip().split("\t")
        data.append(list(map(float, temp)))
    return array(data)

def bin_split_data(data, feature, value):
    data1 = data[data[:,feature] > value,:]
    data2 = data[data[:,feature] <= value,:]
    return data1, data2

def reg_leaf(data):
    return mean(data[:,-1])

def reg_err(data):
    return var(data[:,-1]) * data.shape[0]

def choose_best_split(data, leaf_type, error_type, ops=(1, 4)):
    if len(set(data[:,-1].T.reshape(-1))) == 1:
        return None, leaf_type(data)
    nrow, ncol = data.shape
    Best_error = inf; Best_feature = 0; Best_value = 0
    data_error = error_type(data)
    for i in range(ncol-1):
        for j in data[:,i]:
            arr1, arr2 = bin_split_data(data, i, j)
            if (arr1.shape[0] < ops[1]) or (arr2.shape[0] < ops[1]):
                continue
            temp_error = error_type(arr1) + error_type(arr2)
            if Best_error > temp_error:
                Best_error = temp_error
                Best_feature = i
                Best_value = j
    arr1, arr2 = bin_split_data(data, Best_feature, Best_value)
    if (data_error - Best_error) < ops[0]:
        return None, leaf_type(data)
    if (arr1.shape[0] < ops[1]) or (arr2.shape[0] < ops[1]):
        return None, leaf_type(data)
    return Best_feature, Best_value

def create_tree(data, leaf_type=reg_leaf, error_type=reg_err, ops=(1,4)):
    best_feature, best_value = choose_best_split(data, leaf_type,
                                                 error_type, ops)
    if best_feature is None:
        return best_value
    tree = {}
    tree["spInd"] = best_feature
    tree["spVal"] = best_value
    left_arr, right_arr = bin_split_data(data, best_feature, best_value)
    tree["left"] = create_tree(left_arr, leaf_type, error_type, ops)
    tree["right"] = create_tree(right_arr, leaf_type, error_type, ops)
    return tree

def is_tree(tree):
    return (type(tree).__name__ == "dict")

def get_tree_mean(tree):
    if is_tree(tree["left"]):
        left = get_tree_mean(tree["left"])
    else:
        left = tree["left"]
    if is_tree(tree["right"]):
        right = get_tree_mean(tree["right"])
    else:
        right = tree["right"]
    return (left + right) / 2.0

def prune(tree, test_data):
    if test_data.shape[0] == 0:
        return get_tree_mean(tree)
    if (is_tree(tree["left"])) or (is_tree(tree["right"])):
        arr1, arr2 = bin_split_data(test_data, tree["spInd"], tree["spVal"])
    if is_tree(tree["left"]):
        tree["left"] = prune(tree["left"], arr1)
    if is_tree(tree["right"]):
        tree["right"] = prune(tree["right"], arr2)
    if (not is_tree(tree["left"])) and (not is_tree(tree["right"])):
        arr1, arr2 = bin_split_data(test_data, tree["spInd"], tree["spVal"])
        error_no_merge = sum(power(arr1[:,-1] - tree["left"], 2)) + sum(power(arr2[:,-1] - tree["right"], 2))   
        tree_mean = (tree["left"] + tree["right"]) / 2.0
        error_merge = sum(power(test_data - tree_mean, 2))
        if error_no_merge > error_merge:
            print("merging")
            return tree_mean
        else:
            return tree
    else:
        return tree

def lin_solve(data):
    nrow, ncol = data.shape
    Xmat = mat(ones((nrow, ncol)))
    Xmat[:,1:] = data[:,0:(ncol-1)]
    Ymat = mat(data[:,-1]).T
    xtx = Xmat.T * Xmat
    if linalg.det(xtx) == 0:
        return "Singular Matrix"
    beta = xtx.I * (Xmat.T) * Ymat
    return beta, Xmat, Ymat

def mod_leaf(data):
    beta, Xmat, Ymat = lin_solve(data)
    return beta

def mod_err(data):
    beta, X, Y = lin_solve(data)
    err = sum(power((Y - X * beta), 2))
    return err


def reg_eval(tree, test):
    return float(tree)

def mod_eval(tree, test):
    x = mat(ones((len(test)+1,1)))
    x[1:,] = mat(test).T
    value = x.T * tree
    return value[0,0]

def tree_forecast(tree, test, modeleval = reg_eval):
    if not is_tree(tree):
        return modeleval(tree, test)
    if test[tree["spInd"]] > tree["spVal"]:
        result = tree_forecast(tree["left"], test, modeleval)
    else:
        result = tree_forecast(tree["right"], test, modeleval)
    return result

def forecast_array(tree, test_data, modeleval = reg_eval): #input test_data column array
    num = len(test_data)
    result = ones((num, 1))
    for i in range(num):
        result[i,] = tree_forecast(tree, test_data[i,:], modeleval)
    return result

















