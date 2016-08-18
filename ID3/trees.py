from math import log
import operator

def createDataSet():
	#1，1，yes，前两个是对不同特征值的回答，最后一个是具体类别
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: 
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt
    
#根据给定特征划分，返回去掉这个特征的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #根据给定特征axis,value筛选所有数据,返回的数据都去掉这条特征的值
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
#循环选择最好的划分特征，最终选出一个
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      
    baseEntropy = calcShannonEnt(dataSet) #计算原始总数据类别（yes/no）的熵
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        
        featList = [example[i] for example in dataSet]#不同数据该特征的所有值
        uniqueVals = set(featList)      
        newEntropy = 0.0
        #计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #符合该特征该值的所有数据
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     #计算该数据类别的熵
        infoGain = baseEntropy - newEntropy     
        
        if (infoGain > bestInfoGain):      
            bestInfoGain = infoGain         #计算最好的信息增益
            bestFeature = i
    return bestFeature                     

#没有任何特征后取出yes/no里面最多的返回
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#创建树，相当于分类器训练
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
#遍历树进行数据测试分类
def classify(inputTree,featLabels,testVec):#输入树，测试数据特征（顺序不一定和树的最好标签顺序相同），测试标签的对应值
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename): #存储树，不需要再花大量时间训练
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):  #读取树，不需要再花大量时间训练
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
