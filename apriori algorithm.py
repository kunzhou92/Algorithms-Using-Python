from numpy import *

def load_data():#output list
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def create_c1(dataset):#input list
    c1_list = []
    for row in dataset:
        for item in row:
            if not [item] in c1_list:
                c1_list.append([item])
    return list(map(frozenset, c1_list))

def scan(data_set, ck, min_support):
    each_count = {}
    for data in data_set:
        for each_ck in ck:
            if each_ck.issubset(data):
                each_count[each_ck]= each_count.get(each_ck, 0) + 1
    total = float(len(data_set))
    relist = []
    support_data = {}
    for each_ck in each_count:
        support = each_count[each_ck] / total
        if support >= min_support:
            relist.insert(0, each_ck)
        support_data[each_ck] = support
    return relist, support_data
    
def create_ck(lk):
    num = len(lk)
    if num == 0:
        return []
    new_size = len(lk[0]) + 1
    ck = []
    for i in range(num):
        for j in range(i+1,num):
            temp = lk[i] | lk[j]
            if (len(temp) == new_size) & (not temp in ck):
                ck.append(temp)
    return ck

def apriori(data_list, min_support):
    L = []
    support_data = {}
    data_set = list(map(set, data_list))
    ck = create_c1(data_set)
    lk, sub_support_data = scan(data_set, ck, min_support)
    L.append(lk)
    support_data.update(sub_support_data)
    k = 0
    while len(L[k]) > 0:
        ck = create_ck(lk)
        lk, sub_support_data = scan(data_set, ck, min_support)
        L.append(lk)
        support_data.update(sub_support_data)
        k += 1
    return L, support_data

def cal_conf(freq_set, H, display_list, minConf, support_data):
    prune_list = []
    for item in H:
        confident = support_data[freq_set] / support_data[freq_set-item]
        if confident > minConf:
            print(freq_set-item, "--->", item, "confident: ", confident)
            display_list.append((freq_set-item, item, confident))
            prune_list.append(item)
    return prune_list

def cal_conf2(freq_set, H, display_list, minConf, support_data):
    if (len(H) == 0):
        return
    if (len(freq_set) > len(H[0])):
        prune_list = cal_conf(freq_set, H, display_list, minConf, support_data)
        H2 = create_ck(prune_list)
        if (len(H) > 0):
            cal_conf2(freq_set, H2, display_list, minConf, support_data)

def generate_rules(L, support_data, minConf=0.7):
    num = len(L)
    display_list = []
    for i in range(1,num):
        for freq_set in L[i]:
            H = [frozenset([k]) for k in freq_set]
            cal_conf2(freq_set, H, display_list, minConf, support_data)
    return display_list




















        


