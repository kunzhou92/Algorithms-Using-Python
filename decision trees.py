import math
import operator
import matplotlib.pyplot as plt
def create_dataset():          #   import is none, return is a tuple of lists 
    dataset= [[1, 1, 'yes'],
              [1, 1, 'yes'],
              [1, 0, 'no'],
              [0, 1, 'no'],
              [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    return dataset, label



def calculate_entropy(dataset): #import is a list of lists, return is a float
    row_len = len(dataset)
    dict = {}
    entropy = 0
    for row in dataset:
        label = row[-1]
        dict[label] = dict.get(label, 0) + 1
    for key in dict:
        probability = dict[key] / row_len
        entropy -= probability * math.log(probability, 2)
    return entropy

def split_dataset(dataset, the_ith_feature, value):#dataset is a list, the_ith_feature is a int
    split_data = []
    for row in dataset:
        if row[the_ith_feature] == value:
            row_except_ith_feature = row[:the_ith_feature]
            row_except_ith_feature.extend(row[the_ith_feature+1:])
            split_data.append(row_except_ith_feature)
    return split_data
        
def choose_best_split(dataset):
    feature_len = len(dataset[0]) - 1
    total_len = len(dataset)
    entropy =[]
    for feature_index in range(feature_len):
        feature_value = [row[feature_index] for row in dataset]
        feature_value = list(set(feature_value))
        post_entropy = 0
        for value in feature_value:
            subdataset = split_dataset(dataset, feature_index, value)
            sub_entropy = calculate_entropy(subdataset)
            sub_len = len(subdataset)
            post_entropy += sub_entropy * sub_len / total_len
        entropy.append(post_entropy)
    min_entropy = None
    index = None
    for i in range(len(entropy)):
        if min_entropy is None or min_entropy > entropy[i]:
            min_entropy = entropy[i]
            index = i
    return index

def majority_class(dataset):
    dict = {}
    for row in dataset:
        dict[row[-1]] = dict.get(row[-1], 0) + 1
    index = sorted(dict.items(), key = operator.itemgetter(1), reverse = True)
    return index[0][0]
    
def create_tree(dataset, labels):
    class_list = [row[-1] for row in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(labels) == 0:
        return majority_class(dataset)
    best_split_index = choose_best_split(dataset)
    best_split_label = labels[best_split_index]
    del(labels[best_split_index])
    my_tree = {best_split_label:{}}
    best_feature_value = set([row[best_split_index] for row in dataset])
    for i in best_feature_value:
        sublables = labels[:]
        subdataset = split_dataset(dataset, best_split_index, i)
        my_tree[best_split_label][i] = create_tree(subdataset, sublables)
    return my_tree

decision_node = dict(boxstyle = "sawtooth", fc ="0.8")
leaf_node = dict(boxstyle = "round4", fc = "0.8")

def plot_node(node_text, point_cor, text_cor, nodetype, subfigure):
    subfigure.annotate(node_text, xy = point_cor, xycoords = 'axes fraction',
                       xytext = text_cor, textcoords = 'axes fraction',
                       va = 'center', ha = 'center', bbox = nodetype, 
                       arrowprops = dict(arrowstyle = "->"))

def createplot():
    fig = plt.figure(1, facecolor = 'white')
    fig.clf()
    subplot = plt.subplot(111, frameon = False)
    plot_node("decison", (0.1, 0.5), (0.5, 0.1), decision_node, subplot)
    plot_node("leaf", (0.8, 0.2), (0.2, 0.8), leaf_node, subplot)
    plt.show()

def retrieve_tree(i):
    listoftree = [{"no surfacing":{0:"no", 1:{"fkiooers":{0:"no", 1:"yes"}}}},
                  {"no surfacing":{0:"no", 1:{"flippers": {0:{"head":{0:"no",1:"yes"}},1:"no"}}}}
                  ]
    return listoftree[i]


def get_tree_leafs(tree):
    key = list(tree)[0]
    second_tree = tree[key]
    leafs = 0
    for i in second_tree:
        if type(second_tree[i]).__name__ == "dict":
            leafs += get_tree_leafs(second_tree[i])
        else:
            leafs += 1
    return leafs

def get_tree_depth(tree):
    depth = 0
    key = list(tree)[0]
    second_tree = tree[key]
    for i in second_tree:
        if type(second_tree[i]).__name__ == "dict":
            temp = 1 + get_tree_depth(second_tree[i])
        else:
            temp = 1
        if temp > depth:
            depth = temp
    return depth

def plot_mid_text(point_cor, text_cor, text, subplot):
    x = 0.5 * point_cor[0] + 0.5 * text_cor[0]
    y = 0.5 * point_cor[1] + 0.5 * text_cor[1]
    subplot.text(x, y, text)

def return_x(tree, length):
    key = list(tree)[0]
    second_dict = tree[key]
    sub_length = []
    cumulative_length = []
    each_cumulative_length = 0
    total_leafs = get_tree_leafs(tree)
    for i in second_dict:
        if type(second_dict[i]).__name__ == "dict":
            each_sub_length = get_tree_leafs(second_dict[i]) / total_leafs * length
            
        else:
            each_sub_length = 1 / total_leafs * length
        each_cumulative_length += each_sub_length
        sub_length.append(each_sub_length)
        cumulative_length.append(each_cumulative_length)
    relative_position = []
    for i in range(len(sub_length)):
        if i == 0:
            relative_position.append(cumulative_length[i] / 2 - 0.5 * length)
        else:
            relative_position.append(cumulative_length[i] / 2 + cumulative_length[i-1] /2 - 0.5 * length)         
    return relative_position, sub_length
    
def plot_tree(tree, x, y, subplot, length, length_by_depth):
    key = list(tree)[0]
    second_dict = tree[key]
    seq, sublength = return_x(tree, length)
    count = 0
    plot_node(key, (x, y), (x, y), decision_node, subplot)
    for i in second_dict:
        x_corr = seq[count] + x
        if type(second_dict[i]).__name__ == "dict":
            subplot.annotate("",xy = (x,y), xycoords = 'axes fraction',
                       xytext = (x_corr,y-length_by_depth), textcoords = 'axes fraction',
                       va = 'center', ha = 'center',
                       arrowprops = dict(arrowstyle = "<-"))
            plot_mid_text((x,y), (x_corr,y-length_by_depth), i, subplot)
            plot_tree(second_dict[i], x_corr, y-length_by_depth, subplot, sublength[count],length_by_depth)

        else:
            subplot.annotate(second_dict[i], xy = (x,y), xycoords = 'axes fraction',
                       xytext = (x_corr,y-length_by_depth), textcoords = 'axes fraction',
                       va = 'center', ha = 'center', bbox = leaf_node,
                       arrowprops = dict(arrowstyle = "<-"))
            plot_mid_text((x,y), (x_corr,y-length_by_depth), i, subplot)
        count += 1
            

def create_tree_plot(intree):
    fig = plt.figure(1, facecolor = "white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    subplot = plt.subplot(111,frameon = False, **axprops)
    length_by_depth = 1 / get_tree_depth(intree)
    plot_tree(intree, 0.5, 1.0, subplot, 1.0, length_by_depth)
    plt.show()

def classify(tree, label, test_vector):
    first_str = list(tree)[0]
    index = label.index(first_str)
    second_dict = tree[first_str]
    for item in second_dict:
        if item == test_vector[index]:
            if type(second_dict[item]).__name__ == "dict":
                return classify(second_dict[item], label, test_vector)
            else:
                return second_dict[item]

def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def save_tree(tree, filename):
    fp = open(filename, "wb")
    import pickle
    pickle.dump(tree, fp)
    fp.close()

def load_tree(filename):
    fp = open(filename,"rb")
    import pickle
    return pickle.load(fp)

    



            
        
        
