from numpy import *

def load_data(file_name): #output x matrix, y matrix
    feature_num = len( open(file_name).readline().strip().split('\t') ) - 1
    x = []
    y = []
    for i in open(file_name).readlines():
        temp = i.strip().split('\t')
        temp_list = []
        for j in range(feature_num):
            temp_list.append(float(temp[j]))
        x.append(temp_list)
        y.append(float(temp[-1]))
    return mat(x), mat(y).T

def lin_regression(xmat, ymat): #output matrix
    xtx = xmat.T * xmat
    if linalg.det(xtx) == 0:
        print("error")
        return
    else:
        return linalg.solve(xtx, xmat.T * ymat)

def lwlr(testpoint, xmat, ymat, k=1.0):# testpoint row array  output number
    nrow, ncol = xmat.shape
    weight = mat(eye(nrow))
    xarr = asarray(xmat)
    for i in range(nrow):
        weight[i,i] = exp( sum((xarr[i] - testpoint) ** 2) /(-2.0 * k ** 2) )
    xtwx = xmat.T * weight * xmat
    if linalg.det(xtwx) == 0:
        print("error")
        return
    beta = xtwx.I * xmat.T * weight * ymat
    result = mat(testpoint) * beta
    return result[0,0]

def test_lwlr(test_arr, xmat, ymat, k=1.0): #output matrix
    n = len(test_arr)
    result = mat(zeros((n,1)))
    for i in range(n):
        result[i,0] = lwlr(test_arr[i], xmat, ymat, k)
    return result

def calculate_rss(yhatmat, ymat): #output number
    rss = (ymat - yhatmat).T * (ymat - yhatmat)
    return rss[0,0]

def ridge_regression(xmat, ymat, lam=0.2): #output matrix
    denom = xmat.T * xmat + lam * eye(xmat.shape[1])
    if linalg.det(denom) == 0:
        print("error")
        return
    beta = denom.I * xmat.T * ymat
    return beta

def ridge_test(xmat, ymat):
    y_normalize = ymat - ymat.mean(0)
    x_normalize = (xmat - xmat.mean(0))/sqrt(xmat.var(0))
    num = 30
    beta_mat = mat(zeros((num,x_normalize.shape[1])))
    for i in range(30):
        lam = exp(i-10)
        beta = ridge_regression(x_normalize, y_normalize, lam)
        beta_mat[i,:] = beta.T
    return beta_mat

def stagewise(xmat, ymat, esp=0.01, numit=100):
    ymat = (ymat - ymat.mean(0)) / sqrt(ymat.var(0))
    xmat = (xmat - xmat.mean(0)) / sqrt(xmat.var(0))
    nrow, ncol = xmat.shape
    beta = mat(zeros((ncol,1))); beta_test = beta.copy(); beta_max = beta.copy()
    return_mat = mat(zeros((numit, ncol)))
    for i in range(numit):
        min_error = inf
        for j in range(ncol):
            for sign in [-1, 1]:
                beta_test = beta.copy()
                beta_test[j,0] += sign * esp
                yhat = xmat * beta_test
                rss = calculate_rss(yhat, ymat)
                if rss < min_error:
                    min_error = rss
                    beta_max = beta_test
        return_mat[i,:] = beta_max.T
        beta = beta_max.copy()
    return return_mat









    
        
