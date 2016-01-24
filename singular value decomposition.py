from numpy import *

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]
    
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def ecludSim(inA, inB):
    return 1 / (1 + linalg.norm(inA - inB))

def pearsSim(inA, inB):
    if len(inA) == 1 or (len(list((set(inA.A.reshape(-1))))) == 1 and \
                         len(list((set(inB.A.reshape(-1))))) == 1):
        return 1.0
    elif (len(list((set(inA.A.reshape(-1))))) == 1 or \
                         len(list((set(inB.A.reshape(-1))))) == 1):
        return 0;
    else:
        return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0, 1]

def cosSim(inA, inB):
    value = 0.5 + 0.5 * inA.T * inB / sqrt( linalg.norm(inA) * linalg.norm(inB))
    return value[0,0]

def standEst(dataMat, user, SimMethod, item):
    numUser, numItem = dataMat.shape
    sumSim = 0
    sumRankSim = 0
    for j in range(numItem):
        if dataMat[user, j] == 0:
            continue
        overlap = logical_and(dataMat[:,item] > 0, dataMat[:,j] > 0)
        if any(overlap) == False:
            similarity = 0
        else:
            similarity = SimMethod(dataMat[overlap.A.reshape(-1),item], \
                                    dataMat[overlap.A.reshape(-1),j])
        sumSim += similarity
        sumRankSim += similarity * dataMat[user, j]
    if sumSim == 0:
        return 0
    else:
        result = sumRankSim / sumSim
        return result

def recommend(dataMat, user, N=3, SimMethod = cosSim, estMethod = standEst):
    resultList = []
    for i in range(dataMat.shape[1]):
        if dataMat[user, i] == 0:
            rankValue = estMethod(dataMat, user, SimMethod, i)
            resultList.append((i, rankValue))
    if len(resultList) == 0:
        return "you rated everything"
    if len(resultList) < N:
        return [i for i in sorted(resultList, key = lambda v: v[1],\
                                 reverse = True)]
    newList = [i for i in sorted(resultList, key = lambda v: v[1],\
                                    reverse = True)]
    return newList[:N]

def svdEst(dataMat, user, SimMethod, item):
    U, Sigma, Vt = linalg.svd(dataMat)
    sumSim = 0
    sumRankSim = 0
    numUser, numItem = dataMat.shape
    SigmaDiag = mat(diag(Sigma[:4]))
    newData = dataMat.T * U[:,:4] * SigmaDiag.I
    for j in range(numItem):
        if dataMat[user, j] == 0 or j == item:
            continue
        similarity = SimMethod(newData[item, :].T, newData[j, :].T)
        print("the %d and %d similarity is %f" %(item, j, similarity))
        sumSim += similarity
        sumRankSim += similarity * dataMat[user, j]
    if sumSim == 0:
        return 0
    else:
        result = sumRankSim / sumSim
        return result






            
