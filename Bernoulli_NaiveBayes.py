import pandas as pd
import numpy as np
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer

train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']

comments = train['comments']
labels = train['subreddits']

# Process Data into train set and test set
C_train, C_test, L_train, L_test = train_test_split(comments, labels, random_state = 0)

#conclude comments into a [V] vocabulary vector
def getVocabularyVector (C_train):
    dataSet = C_train.copy()

    vocabularyVector = np.array([])
    return vocabularyVector

#preprocess comments into a two-dimentional binary matrix based on the absence and presence of words in [V]
def preprocessComments (vocabV , C_train):

    documentM=np.array([])
    return documentM

def fit (vocabV,documM,L_train,C_train):
    #total number of comments
    N=C_train.shape[0]
    #count number of comments labelled with class K
    Karray = np.array([["hockey",0,[]],["nba",0,[]],["soccer",0,[]],["baseball",0,[]],["GlobalOffensive",0,[]],
                       ["canada",0,[]],["conspiracy",0,[]],["europe",0,[]],["anime",0,[]],["Overwatch",0,[]],
                       ["wow",0,[]],["nfl",0,[]],["leagueoflegends",0,[]],["trees",0,[]],["Music",0,[]],
                       ["AskReddit",0,[]],["worldnews",0,[]],["funny",0,[]],["gameofthrones",0,[]],["movies",0,[]]])
    for x in range(L_train.shape[0]):
        for y in range(Karray.shape[0]):
            if(L_train[x]==Karray[y][0]):
                Karray[y][1]+=1
                Karray[y][2].append(documM[x])

    numberOfCommentsContainWordInClass=[[0]*vocabV.shape[0]]*Karray.shape[0]
    #count number of comments of class K containing word w
    for i in range(Karray.shape[0]):
        index=0
        for l in range(Karray[i][2].shape[0]):
            if(Karray[i][2][l][index]==1):
                numberOfCommentsContainWordInClass[i][l]+=1
        index+=1

    #compute the relative frequency of comments of class K
    totalNumberOfComments = C_train.shape[0]
    priors=[0]*Karray.shape[0]
    for p in range(Karray.shape[0]):
        priors[p]=Karray[p][1]/totalNumberOfComments

    #compute probabilities of each word given the comment class
    likelyhoods=[[0]*vocabV.shape[0]]*Karray.shape[0]
    for q in range(Karray.shape[0]):
        for s in range(vocabV.shape[0]):
            likelyhoods[q][s]=numberOfCommentsContainWordInClass[q][s]/Karray[q][1]

    result = [priors,likelyhoods]
    return result


#To classify an unlabelled comment in C_test,we estimate the posterior probability for each class K
def predict (priors,likelyhoods,C_test):
    Karray = np.array(
        ["hockey", "nba", "soccer", "baseball", "GlobalOffensive",
         "canada", "conspiracy", "europe", "anime", "Overwatch",
         "wow", "nfl", "leagueoflegends", "trees", "Music",
         "AskReddit", "worldnews", "funny", "gameofthrones", "movies"])
    vocVector=getVocabularyVector(C_test)
    docMatrix=preprocessComments(vocVector,C_test)
    for i in range(docMatrix.shape[0]):
        for j in range(docMatrix.shape[1]):
            if (docMatrix[i][j]==1):
                docMatrix[i][j]*likelyhoods[i][j]
            else:
                docMatrix[i][j]=1-likelyhoods[i][j]

    product=[0]*20
    for p in range(docMatrix.shape[0]):
        for q in range(docMatrix.shape[1]):
            product[p]=product[p]*docMatrix[p][q]


    result = Karray[0]
    return result