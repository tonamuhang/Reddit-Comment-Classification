import pandas as pd
import numpy as np
import math
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
from nltk import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from string import punctuation
import re

nltk.download('punkt')

# Read in files
train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')

comments = train['comments']
test_x = test['comments']
labels = train['subreddits']

#read preprocessed training data from the train_processed.csv file
#to avoid the waste of time, we preprocessed both training data and test data in advance to be analysed
keywords = pd.read_csv("train_processed.csv", header=None).astype(str)
keywords = keywords[keywords.columns[1]]
keywords = keywords.apply(word_tokenize)

#resd preprocessed test data from the test_processed.csv file
testdata = pd.read_csv("test_processed.csv", header=None).astype(str)
testdata = testdata[testdata.columns[1]]
testdata = testdata.apply(word_tokenize)

labels = labels.iloc[0:-1].tolist()


def removeDuplicateWords(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# conclude comments into a [V] vocabulary vector
def getVocabularyVector():
    vocabularyVector = list()
    for element in keywords:
        vocabularyVector = vocabularyVector + element
    return vocabularyVector


# preprocess comments into a two-dimentional binary matrix based on the absence and presence of words in [V]
def preprocessComments(vocabV,keywords):
    documentM = keywords
    binaryM = np.zeros((documentM.shape[0], len(vocabV)))
    for x in range(len(binaryM)):
        for y in range(len(vocabV)):
            for z in range(len(documentM[x])):
                if (vocabV[y] == documentM[x][z]):
                    binaryM[x][y] = 1
    return binaryM


def fit(vocabV,keywords):
    binaryM = preprocessComments(vocabV,keywords)
    documM = keywords
    # total number of comments
    N = documM.shape[0]
    # count number of comments labelled with class K
    Karray = [["hockey", 0, []], ["nba", 0, []], ["soccer", 0, []], ["baseball", 0, []], ["GlobalOffensive", 0, []],
              ["canada", 0, []], ["conspiracy", 0, []], ["europe", 0, []], ["anime", 0, []], ["Overwatch", 0, []],
              ["wow", 0, []], ["nfl", 0, []], ["leagueoflegends", 0, []], ["trees", 0, []], ["Music", 0, []],
              ["AskReddit", 0, []], ["worldnews", 0, []], ["funny", 0, []], ["gameofthrones", 0, []], ["movies", 0, []]]

    for x in range(len(labels)):  # comments and its tags
        for y in range(len(Karray)):  # 20 subreddits
            if (labels[x] == Karray[y][0]):  # at index [y][0] means subreddit like "hockey"
                Karray[y][1] += 1  # if comments'tag matches Karray [y][0], use the counter [y][1] to count it
                Karray[y][2].append(binaryM[x])  # then, append corresponding binary vector into index[2]

    numberOfCommentsContainWordInClass = []
    for i in range(len(Karray)):
        numberOfCommentsContainWordInClass.append([])
        for j in range(len(vocabV)):
            numberOfCommentsContainWordInClass[i].append([0])

    # count number of comments of class K containing word w
    for i in range(len(Karray)):  # 20 subreddits
        for l in range(len(Karray[i][2])):  # length of the corresponding binary vectors
            for k in range(len(vocabV)):  # 2078 (words)
                if Karray[i][2][l][k] == 1:  # every binary vectors' every word is 1(present)
                    numberOfCommentsContainWordInClass[i][k][0] += 1  #

    # compute the relative frequency of comments of class K
    totalNumberOfComments = N
    priors = []
    for i in range(len(Karray)):
        priors.append([0])
    for p in range(len(Karray)):
        priors[p][0] = Karray[p][1] / totalNumberOfComments

    # compute probabilities of each word given the comment class
    likelihoods = []
    for i in range(len(Karray)):
        likelihoods.append([])
        for j in range(len(vocabV)):
            likelihoods[i].append([0])

    for q in range(len(Karray)):
        for s in range(len(vocabV)):
            likelihoods[q][s][0] = numberOfCommentsContainWordInClass[q][s][0] / Karray[q][1]

    result = [priors, likelihoods]
    return result


# To classify an unlabelled comment in C_test,we estimate the posterior probability for each class K
def predict(testdata):
    Keyarray = ["hockey", "nba", "soccer", "baseball", "GlobalOffensive",
                "canada", "conspiracy", "europe", "anime", "Overwatch",
                "wow", "nfl", "leagueoflegends", "trees", "Music",
                "AskReddit", "worldnews", "funny", "gameofthrones", "movies"]

    vocVector = getVocabularyVector()

    vocVector = removeDuplicateWords(vocVector)

    docMatrix = preprocessComments(vocVector,keywords)

    priors = fit(vocVector,keywords)[0]
    likelyhoods = fit(vocVector,keywords)[1]

    #compute likelihoods of each vocabulary based on its existence and its own likelihoods
    for i in range(docMatrix.shape[0]):
        for j in range(docMatrix.shape[1]):
            for k in range(len(Keyarray)):
                if (labels[i] == Keyarray[k]):
                    if (docMatrix[i][j] == 1):
                        docMatrix[i][j] = docMatrix[i][j] * likelyhoods[k][j][0]
                    else:
                        docMatrix[i][j] = 1 - likelyhoods[k][j][0]

    #By making Naive Bayesm Assumption, we assume that the probability of each word occuring in the document is independant of the occurences of the other words
    #Thus, we multiply then occurences of each word in a class, also known as the multiplication of individual word likelihoods
    product = []
    for j in range(docMatrix.shape[0]):
        product.append([1])

    for p in range(docMatrix.shape[0]):
        for q in range(docMatrix.shape[1]):
            if (docMatrix[p][q] != 0):
                product[p][0] = product[p][0]*docMatrix[p][q]

    # compute posterior probabilities for each comment based on 20 classes
    # the final prediction will be the class that result in the max of all the posterior probabilities
    posteriorProbability = []
    for i in range(len(product)):
        posteriorProbability.append([])
        for j in range(len(priors)):
            posteriorProbability[i].append([0])

    #multiply together all words likelihoods in each comment and then multiply the results with priors probability
    #in order to gain posterior probability
    for x in range(len(product)):
        for y in range(len(priors)):
            posteriorProbability[x][y][0] = product[x][0] * priors[y][0]

    predictionArray = []
    for k in range(len(product)):
        predictionArray.append([0])

    #list variable to store the maximum posterior probability of each row
    maximumPosteriorProbabilities=[]
    for a in range(len(product)):
        maximumPosteriorProbabilities.append([0])

    #by locating the index of the maximum posterior probability on each row
    #we located the index in keyarray such that the comments has the biggest probability being nin that class
    for z in range(len(posteriorProbability)):
        maximunOfRow = 0
        for k in range(len(posteriorProbability[z])):
            if(posteriorProbability[z][k][0]>maximunOfRow):
                maximunOfRow=posteriorProbability[z][k][0]
        maximumPosteriorProbabilities[z][0]=maximunOfRow



    #list variable that stores all the indexies of the maximum posterior probability of each row
    indexOfMaxima = []
    for k in range(len(product)):
        indexOfMaxima.append([0])

    #for loop that gets the indexies of the maximum posterior probabilities of each row
    #Meanwhile, it gets the corresponding indexies in the keyarray which is the anticipated prediction class
    for k in range (len(indexOfMaxima)):
        indexOfMaxima[k][0] = posteriorProbability[k].index(maximumPosteriorProbabilities[k][0])
        predictionArray[k][0] = Keyarray[indexOfMaxima[k][0]]

    return predictionArray



print(*predict(testdata), sep=",")

