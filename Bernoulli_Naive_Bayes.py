import sys
import os
import re
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import string
import copy

def populateStopWords():
    stopWords = []
    with open('stopwords.txt') as inputFile:
        for line in inputFile:
            if(line != '\n'):
                stopWords.append(line.rstrip())
    return stopWords

def tokenize(inputString):
    whitespaceStripped = re.sub("\s+", " ", inputString.strip())
    punctuationRemoved = "".join([x for x in whitespaceStripped
                                  if x not in string.punctuation])
    lowercase = punctuationRemoved.lower()
    return lowercase.split()

#split a whole file to single word in a list
def splitFile(fileName):
    words = []
    with open(fileName, errors='ignore') as inputFile:
        for line in inputFile:
            words.extend(tokenize(line))
    return words

def computeConfusionMatrix(predicted, groundTruth, nAuthors):
    confusionMatrix = [[0 for i in range(nAuthors+1)] for j in range(nAuthors+1)]
    for i in range(len(groundTruth)):
        confusionMatrix[predicted[i]][groundTruth[i]] += 1
    return confusionMatrix

def outputConfusionMatrix(confusionMatrix):
    columnWidth = 4
    print(str(' ').center(columnWidth),end=' ')
    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')
    print()
    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')
        for j in range(1,len(confusionMatrix)):
            print(str(confusionMatrix[j][i]).center(columnWidth),end=' ')
        print()

#get list of answer from the list of filename
def get_answer(fileNameList):
    test_ground_truth = []
    with open('test_ground_truth.txt', errors='ignore') as inputFile:
        for line in inputFile:
            test_ground_truth.append((line.split()))
    answer = []
    for fileName in fileNameList:
        for line in test_ground_truth:
            if line == []:
                continue
            if line[0][-21:] == fileName[-21:]:
                authorAnswer = line[1][6:]
                if authorAnswer == '__':
                    answer.append(-1)
                else:               
                    answer.append(int(authorAnswer))
                break
    return answer

#calculate accuracy from the result and answer
def get_accuracy(result, answer):
    total = 0
    rightAnswer = 0
    for i in range(len(result)):
        if result[i] == answer[i]:
            rightAnswer += 1
        total += 1
    return rightAnswer/total

#train the trainSet using the vocabulary
def train(trainSet, vocabulary):
    prior = []
    condProb = []
    totalTrainFile = 0;
    for trainAuthorSet in trainSet:
        totalTrainFile += len(trainAuthorSet)

    for trainAuthorSet in trainSet:
        docsOfAuthor = len(trainAuthorSet)
        prior.append(docsOfAuthor/totalTrainFile)
        condProbAuthor = []
        for feature in vocabulary:
            docsWithFeature = 0
            for document in trainAuthorSet:
                if feature in document:
                    docsWithFeature += 1
            condProbAuthor.append((docsWithFeature+1)/(docsOfAuthor+2))
        condProb.append(condProbAuthor.copy())
    return prior, condProb

#test the sampleSet using the result of train
def apply(vocabulary, prior, condProb, sampleSet):
    result = []
    for sample in sampleSet:
        score = []
        score = prior.copy()
        for i in range(len(score)):
            score[i] = math.log2(score[i])
            for j in range(len(vocabulary)):
                if vocabulary[j] in sample:
                    score[i] += math.log2(condProb[i][j])
                else:
                    score[i] += math.log2(1-condProb[i][j])
        result.append(1+score.index(max(score)))
    return result

#read in from the path and get the trainSet sampleSet and vocalulary
def read(inputPath):
    problemSetCharacter = inputPath[-2]
    inputPath = inputPath + problemSetCharacter
    fileNameListSample = []
    fileIndex = 1
    while (os.path.isfile(inputPath + 'sample' +str(fileIndex).zfill(2)+ '.txt')):
        fileNameListSample.append(inputPath + 'sample' +str(fileIndex).zfill(2)+ '.txt')
        fileIndex += 1
    sampleSet = []
    for fileNameSample in fileNameListSample:
        sampleSet.append(splitFile(fileNameSample))
    sampleAnswer = get_answer(fileNameListSample)
    fileLabelIndex = 1
    fileTrainIndex = 1
    trainSet = [[]]
    while True:
        tempStringOne = inputPath + 'train' + str(fileLabelIndex).zfill(2) + '-' + str(fileTrainIndex).zfill(1) + '.txt'
        tempStringTwo = inputPath + 'train' + str(fileLabelIndex).zfill(2) + '-' + str(fileTrainIndex).zfill(2) + '.txt'
        tempNextStringOne = inputPath + 'train' + str(fileLabelIndex).zfill(2) + '-' + str(fileTrainIndex+1).zfill(1) + '.txt'
        tempNextStringTwo = inputPath + 'train' + str(fileLabelIndex).zfill(2) + '-' + str(fileTrainIndex+1).zfill(2) + '.txt'
        if os.path.isfile(tempStringOne):
            trainSet[-1].append(splitFile(tempStringOne))
            fileTrainIndex += 1
        elif os.path.isfile(tempStringTwo):
            trainSet[-1].append(splitFile(tempStringTwo))
            fileTrainIndex += 1
        else :
            if fileTrainIndex == 1:
                if os.path.isfile(tempNextStringOne) or os.path.isfile(tempNextStringTwo):
                    fileTrainIndex += 1
                else :
                    break
            else :
                if os.path.isfile(tempNextStringOne) or os.path.isfile(tempNextStringTwo):
                    fileTrainIndex += 1
                else :
                    fileLabelIndex += 1
                    fileTrainIndex = 1
                    trainSet.append([])
    del trainSet[-1]
    return trainSet, sampleSet, sampleAnswer
    
#calculate CCE from the index of feature
def getCCE(prior, condProb, index):
    CCE = 0
    for i in range(len(prior)):
        CCE -= (prior[i]*condProb[i][index]*math.log2(condProb[i][index]))
    return CCE

#calculate feature ranking from the vocalulary and the conditional probilities
def getFeatureRank(prior, condProb, vocabulary):
    featureCCE = []
    for i in range(len(vocabulary)):
        featureCCE.append((vocabulary[i], getCCE(prior, condProb, i)))
    featureCCE.sort(key=lambda x: x[1], reverse=True)
    return featureCCE[:20]

#calculate the feature in vocabulary appear times in trainSet
#and sort them in descending order
def getFrequency(trainSet, vocabulary):
    featureFrequency = []
    for feature in vocabulary:
        frequency = 0
        for trainAuthorSet in trainSet:
            for document in trainAuthorSet:
                frequency += document.count(feature)
        featureFrequency.append((feature, frequency))
    featureFrequency.sort(key=lambda x: x[1], reverse=True)
    return featureFrequency

def main():
    stopWords = populateStopWords()
    trainSet, sampleSet, answer = read(sys.argv[1])
    vocabulary = sorted(list(set(stopWords)))
    prior, condProb = train(trainSet, vocabulary)
    result = apply(vocabulary, prior, condProb, sampleSet)
    accuracy = get_accuracy(result, answer)
    accuracyList = []
    print('-----------------Test Accuracy------------------')
    print('Accuracy: ' + str(accuracy))
    print()
    print('----------------Confusion Matrix----------------')
    outputConfusionMatrix(computeConfusionMatrix(result, answer, len(prior)))
    print()
    featureRank = getFeatureRank(prior, condProb, vocabulary)
    print('----------------Feature Ranking-----------------')
    print(featureRank)
    print()
    print('-----------------Feature Curve------------------')
    featureFrequency = getFrequency(trainSet, vocabulary)
    featureNumber = []
    for i in range(math.floor(len(vocabulary)/10)):
        vocabulary,_ =  zip(*featureFrequency[:10*(i+1)])
        prior, condProb = train(trainSet, vocabulary)
        result = apply(vocabulary, prior, condProb, sampleSet)
        accuracy = get_accuracy(result, answer)
        print('Number of features: '+str(10*(i+1))+'     Accuracy: '+str(accuracy))
        accuracyList.append(accuracy)
        featureNumber.append(10*(i+1))
    plt.figure()
    plt.plot(featureNumber, accuracyList)
    plt.savefig(sys.argv[1]+'Feature_Curve.png')

    print('See '+sys.argv[1]+'Feature_Curve.png for the diagram of Feature Curve')

if __name__ == '__main__':
    main()
