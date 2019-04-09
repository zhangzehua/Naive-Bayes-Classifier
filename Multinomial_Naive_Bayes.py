import sys
import os
import math
import string
import copy

#computer confusion matrix using predict result and answer
def computeConfusionMatrix(predicted, groundTruth, nAuthors):
    confusionMatrix = [[0 for i in range(nAuthors+1)] for j in range(nAuthors+1)]
    for i in range(len(groundTruth)):
        confusionMatrix[predicted[i]][groundTruth[i]] += 1
    return confusionMatrix

#output the confusion matrix
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

#split the file in to list of single words
def split(documentName, characters):
	words = []
	with open(documentName, encoding="iso-8859-15", errors='ignore') as inputFile:
		for line in inputFile:
			for i in range(len(line)-characters):
				words.append(line[i:i+characters].lower())
	return words

#Read from the Train and Test set
def read(inputPath, characters):
	trainNameEnglish = inputPath+'train2000-English.en'
	trainNameFrench = inputPath+'train2000-French.fr'
	trainNameGerman = inputPath+'train2000-German.de'
	trainNameSpanish = inputPath+'train2000-Spanish.es'
	testNameEnglish = inputPath+'test-English.txt'
	testNameFrench = inputPath+'test-French.txt'
	testNameGerman = inputPath+'test-German.txt'
	testNameSpanish = inputPath+'test-Spanish.txt'
	if not os.path.isfile(trainNameEnglish):
		trainNameEnglish  += '.txt'
	if not os.path.isfile(trainNameFrench):
		trainNameFrench  += '.txt'
	if not os.path.isfile(trainNameGerman):
		trainNameGerman  += '.txt'
	if not os.path.isfile(trainNameSpanish):
		trainNameSpanish  += '.txt'
	trainSet = []
	trainSet.append(split(trainNameEnglish, characters))
	trainSet.append(split(trainNameFrench, characters))
	trainSet.append(split(trainNameGerman, characters))
	trainSet.append(split(trainNameSpanish, characters))
	testSet = []
	testSet.append(split(testNameEnglish, characters))
	testSet.append(split(testNameFrench, characters))
	testSet.append(split(testNameGerman, characters))
	testSet.append(split(testNameSpanish, characters))
	vocabulary = []
	for trainLabel in trainSet:
		vocabulary.extend(trainLabel)
	vocabulary = sorted(list(set(vocabulary)))
	return trainSet,testSet,vocabulary

#calculate the feature in vocabulary appear times in trainSet
#and sort them in descending order
def getFrequency(feature, document):
	totalTimes = 0
	for word in document:
		if feature == word:
			totalTimes += 1
	return totalTimes

#train the trainSet using the vocabulary
def train(trainSet, vocabulary):
	condProb = []
	v_len = len(vocabulary)
	dict = {}
	for i in range(v_len):
		dict[vocabulary[i]] = i
	for trainLabel in trainSet:
		condProbLanguage = [0]*v_len
		t_len = len(trainLabel)
		for word in trainLabel:
			condProbLanguage[dict[word]] += 1
		for i in range(v_len):
			condProbLanguage[i] = (condProbLanguage[i]+1)/(t_len+v_len)
		condProb.append(condProbLanguage.copy())
	prior = [1/4, 1/4, 1/4, 1/4]
	return condProb, prior

#test the sampleSet using the result of train
def test(testSet, condProb, prior, vocabulary):
	result = []
	for testLabel in range(len(testSet)):
		score = prior.copy()
		for i in range(len(prior)):
			score[i] = math.log2(score[i])
			for word in testSet[testLabel]:
				if word in vocabulary:
					score[i] += math.log2(condProb[i][vocabulary.index(word)])
		result.append(1+score.index(max(score)))
	return result

#calculate accuracy from the result and answer
def get_accuracy(result, answer):
    total = 0
    rightAnswer = 0
    for i in range(len(result)):
        if result[i] == answer[i]:
            rightAnswer += 1
        total += 1
    return rightAnswer/total

#Read, Train, Test, Output
def main():
	answer = [1, 2, 3, 4]
	trainSet,testSet,vocabulary = read(sys.argv[1],2)
	print(len(vocabulary))
	condProb, prior = train(trainSet, vocabulary)
	bigramResult = test(testSet, condProb, prior, vocabulary)
	bigramAccuracy = get_accuracy(bigramResult, answer)
	trainSet,testSet,vocabulary = read(sys.argv[1],3)
	print(len(vocabulary))
	condProb, prior = train(trainSet, vocabulary)
	trigramResult = test(testSet, condProb, prior, vocabulary)
	trigramAccuracy = get_accuracy(trigramResult, answer)
	print()
	print('                    Bigram                      ')
	print('-----------------Test Accuracy------------------')
	print('Accuracy: ' + str(bigramAccuracy))
	print()
	print('----------------Confusion Matrix----------------')
	outputConfusionMatrix(computeConfusionMatrix(bigramResult, answer, len(prior)))
	print()
	print()
	print()
	print('                    Trigram                     ')
	print('-----------------Test Accuracy------------------')
	print('Accuracy: ' + str(trigramAccuracy))
	print()
	print('----------------Confusion Matrix----------------')
	outputConfusionMatrix(computeConfusionMatrix(trigramResult, answer, len(prior)))
	print()


if __name__ == '__main__':
    main()