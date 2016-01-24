import numpy

def load_dataset():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def create_vocabulary_list(dataset):
    vocabulary_set = set([])
    for row in dataset:
        vocabulary_set = vocabulary_set | set(row)
    return list(vocabulary_set)

def normalize_data(input_data, vocabulary_list):
    set_count = [0] * len(vocabulary_list)
    for word in input_data:
        if word in vocabulary_list:
            set_count[vocabulary_list.index(word)] += 1
        else:
            print("%s is not in the list" % word)
    return set_count

def calculate_probability(transformed_data, classlist):
    p_abuse = sum(classlist) / len(classlist)
    column_num = len(transformed_data[0])
    record_0 = numpy.ones(column_num)
    record_1 = numpy.ones(column_num)
    total_0 = 2.0
    total_1 = 2.0
    for i,row in enumerate(transformed_data):
        if classlist[i] == 0:
            record_0 += row
            total_0 += sum(row)
        elif classlist[i] ==1:
            record_1 += row
            total_1 += sum(row)
    return record_0/total_0, record_1/total_1, p_abuse

def classsify(test, p_0, p_1, p_abuse):
    bayes_p_0 = sum(test * numpy.log(p_0)) + numpy.log(1 - p_abuse)
    bayes_p_1 = sum(test * numpy.log(p_1)) + numpy.log(p_abuse)
    if bayes_p_0 > bayes_p_1:
        return 0
    else:
        return 1

def testingNB():
    data, class_list = load_dataset()
    vocabulary_list = create_vocabulary_list(data)
    tran_data = []
    for i in data:
        tran_data.append(normalize_data(i, vocabulary_list))
    p0, p1, p_a = calculate_probability(numpy.array(tran_data), class_list)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(normalize_data(testEntry, vocabulary_list))
    print(testEntry,'classified as: ',classsify(thisDoc,p0,p1,p_a) )
    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(normalize_data(testEntry, vocabulary_list))
    print(testEntry,'classified as: ',classsify(thisDoc,p0,p1,p_a) )   

def parse_txt(string):
    import re
    list_of_tokens = re.split("\W*", string)
    return [i for i in list_of_tokens if len(i) > 2]

def spam_test():
    import random
    doclist = []
    class_list = []
    for i in range(1, 26):
        name = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch04\\email\\ham\\%d.txt" % i
        each_list = parse_txt(open(name).read())
        doclist.append(each_list)
        class_list.append(0)
        name = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch04\\email\\spam\\%d.txt" % i
        each_list = parse_txt(open(name).read())
        doclist.append(each_list)
        class_list.append(1)
    vocabulary_list = create_vocabulary_list(doclist)
    normalize_dataset = []
    for i in range(50):
        normalize_dataset.append(normalize_data(doclist[i], vocabulary_list))
    testing_samples = []
    for i in range(10):
        index = int(random.uniform(0, len(normalize_dataset)))
        testing_samples.append(normalize_dataset[index])
        del(normalize_dataset[index])
        del(class_list[index])
    p0, p1, pa = calculate_probability(numpy.array(normalize_dataset), class_list)
    error = 0
    for i in range(10):
        if class_list[i] != classsify(testing_samples[i], p0, p1, pa):
            error += 1
    return error

def sort_most_freq(full_text, vocabulary_list):
    import operator
    text_dict = {}
    for i in full_text:
        text_dict[i] = text_dict.get(i, 0) + 1
    sort_list = sorted(text_dict, key = operator.itemgetter(1), reverse = True)
    return sort_list[:40]
































    
    
    
