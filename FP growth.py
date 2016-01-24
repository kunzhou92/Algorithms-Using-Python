from numpy import *
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.parent = parentNode
        self.nodeLink = None
        self.children = {}
    def inc(self, num):
        self.count += num
    def disp(self, integer=1):
        print(" "*integer, self.name, " ", self.count)
        integer += 1
        for child in self.children.values():
            child.disp(integer)
        
def createTree(data, minSup=1):
    headerTable={}
    for itemSet in data:
        for item in itemSet:
            headerTable[item] = headerTable.get(item, 0) + data[itemSet]
    for item in list(headerTable.keys()):
        if headerTable[item] < minSup:
            del(headerTable[item])
        else:
            headerTable[item] = [headerTable[item], None]
    if len(headerTable) == 0:
        return None, None
    Tree = treeNode("Null set", 0, None)
    for itemSet, count in data.items():
        temp = {}
        for item in itemSet:
            if item in headerTable:
                temp[item] = headerTable[item][0]
        orderSet = [i[0] for i in sorted(temp.items(),\
                                    key = lambda k: k[1], reverse = True)]
        if len(orderSet) > 0:
            updateTree(orderSet, Tree, headerTable, count)
    return Tree, headerTable

def updateTree(Set, Tree, headerTable, count):
    if len(Set) > 0:
        if Set[0] in Tree.children:
            Tree.children[Set[0]].inc(count)
        else:
            Tree.children[Set[0]] = treeNode(Set[0], count, Tree)
            updateHeader(Tree.children[Set[0]], headerTable)
        updateTree(Set[1:], Tree.children[Set[0]], headerTable, count)

def updateHeader(Tree, headerTable):
    if headerTable[Tree.name][1] is None:
        headerTable[Tree.name][1] = Tree
    else:
        point = headerTable[Tree.name][1]
        while(point.nodeLink != None):
            point = point.nodeLink
        point.nodeLink = Tree
        
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def ascendTree(prefixPath, Tree):
    if not Tree.parent is None:
        prefixPath.append(Tree.name)
        ascendTree(prefixPath, Tree.parent)

def findPrefixpath(item, Tree):
    record = {}
    while(not Tree is None):
        prefixPath = []
        ascendTree(prefixPath, Tree)
        if len(prefixPath) > 1:
            record[frozenset(prefixPath[1:])] = Tree.count
        Tree = Tree.nodeLink
    return record

def mineTree(Tree, headerTable, minSupport, prefix, freqItemList):
    sortedHeader = [i[0] for i in \
                    sorted(headerTable.items(), key = lambda k: k[1][0])]
    for basePat in sortedHeader:
        newFreqSet = prefix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBase = findPrefixpath(basePat, headerTable[basePat][1])
        subTree, subHeader = createTree(condPattBase, minSupport)
        if subHeader != None:
            mineTree(subTree, subHeader, minSupport, newFreqSet, freqItemList)

#filename = "E:\\Anaconda3\\SourceCode\\machinelearninginaction\\Ch12\\kosarak.dat"
#data = [line.split() for line in open(filename).readlines()]





















    



