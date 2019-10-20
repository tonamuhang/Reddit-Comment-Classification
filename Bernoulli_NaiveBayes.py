import pandas as pd
import numpy as np
import math
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
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

# Show first 5 rows of raw data

# # Remove URLs
# def remove_URLs(text):
#     text = re.sub(r"http\S+", " ", text)
#     return text
#
# # Set all words to lowercase
# def set_lowercase(text):
#     text = text.lower();
#     return text;
#
# # Remove tags
# def remove_tags(text):
#     text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
#     return text
#
# # Remove special characters and digits
# def remove_special_chars_digits(text):
#     text = re.sub("(\\d|\\W)+"," ",text)
#     return text
#
# # Note: Order of remove operations matters!
# comments = comments.apply(lambda x: remove_URLs(x))
# comments = comments.apply(lambda x: set_lowercase(x))
# comments = comments.apply(lambda x: remove_tags(x))
# comments = comments.apply(lambda x: remove_special_chars_digits(x))
# print("----remove urls tags, set lowercase, remove specials characters and digits----\n", comments.head(5))
#
# # Tokenize comments
# comments = comments.apply(word_tokenize)
# print("----tokenized----\n", comments.head(12))
#
# # Lemmatize
# lemmatizer = nltk.WordNetLemmatizer()
# def lemmatize(text):
#     for row_i in range(comments.shape[0]):
#         for word_j in range(len(comments[row_i])):
#             comments[row_i][word_j] = lemmatizer.lemmatize(comments[row_i][word_j])
#     return text
# comments = lemmatize(comments)
# print("----lemmatized----\n", comments.head(12))
#
# # Remove Stopwords
# stop_words = set(stopwords.words('english'))
# comments = comments.apply(lambda x: [item for item in x if item not in stop_words])
# print("----removed stop words----\n", comments.head(5))
#
# # Repeat Replacer
# # http://www.ishenping.com/ArtInfo/971959.html
# class RepeatReplacer():
#     def __init__(self):
#         self.repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)')
#         self.repl = r'\1\2\3'
#     def replace(self, word):
#         if wordnet.synsets(word):  # 判断当前字符串是否是单词
#             return word
#         repl_word = self.repeat_reg.sub(self.repl, word)
#         if repl_word != word:
#             return self.replace(repl_word)
#         else:
#             return repl_word
# replacer = RepeatReplacer()
# def remove_repeat(text):
#     for row_i in range(comments.shape[0]):
#         for word_j in range(len(comments[row_i])):
#             comments[row_i][word_j] = replacer.replace(comments[row_i][word_j])
#     return text
# comments = remove_repeat(comments)
# print("----removed repeat----\n", comments.head(5))


keywords = pd.read_csv("test_processed_oct20.csv",header = None).astype(str)
keywords = keywords[keywords.columns[1]]
keywords = keywords.apply(word_tokenize)
print("```````start`````````")
print("-keywords-")
print(keywords)
print("``````````````````````````````````")
print("keywords[0]", keywords[0])
print("keywords[0] length", len(keywords[0]))
print("``````````````````````````````````")
labels=labels.iloc[0:201].tolist()
print("labels", labels)
print('labels type is list')
print("labels length", len(labels))
print("``````````````````````````````````")

def removeDuplicateWords(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


#conclude comments into a [V] vocabulary vector
def getVocabularyVector ():
    vocabularyVector=list()
    for element in keywords:
        vocabularyVector=vocabularyVector+element
    return vocabularyVector

#preprocess comments into a two-dimentional binary matrix based on the absence and presence of words in [V]
def preprocessComments (vocabV):
    documentM = keywords
    binaryM = np.zeros((documentM.shape[0],len(vocabV)))
    for x in range(len(binaryM)):
        for y in range(len(vocabV)):
            for z in range(len(documentM[x])):
                if(vocabV[y]==documentM[x][z]):
                    binaryM[x][y]=1
    return binaryM

def fit (vocabV):
    # vocabV has length 2078
    print("In fit function")
    binaryM = preprocessComments(vocabV)
    documM = keywords
    #total number of comments
    N=documM.shape[0]
    #count number of comments labelled with class K
    Karray = [["hockey",0, []],     ["nba",0,[]],["soccer",0,[]],["baseball",0,[]],["GlobalOffensive",0,[]],
                       ["canada",0,[]],["conspiracy",0,[]],["europe",0,[]],["anime",0,[]],["Overwatch",0,[]],
                       ["wow",0,[]],["nfl",0,[]],["leagueoflegends",0,[]],["trees",0,[]],["Music",0,[]],
                       ["AskReddit",0,[]],["worldnews",0,[]],["funny",0,[]],["gameofthrones",0,[]],["movies",0,[]]]

    for x in range(len(labels)):  # 201 comments and its tags
        for y in range(len(Karray)): # 20 subreddits
            if(labels[x]==Karray[y][0]): # at index [y][0] means subreddit like "hockey"
                Karray[y][1]+=1          # if comments'tag matches Karray [y][0], use the counter [y][1] to count it
                Karray[y][2].append(binaryM[x]) # then, append corresponding binary vector into index[2]
                #print(len(Karray))
                #print(Karray)

    numberOfCommentsContainWordInClass=[[0]*len(vocabV)]*len(Karray) # (20,2078) 20 lists, every list has 2078 words
    numberOfCommentsContainWordInClass = []
    for i in range(len(Karray)):
        numberOfCommentsContainWordInClass.append([])
        for j in range(len(vocabV)):
            numberOfCommentsContainWordInClass[i].append([0])

    print("numberOfCommentsContainWordInClass", numberOfCommentsContainWordInClass)
    #count number of comments of class K containing word w


    for i in range(len(Karray)):              # 20 subreddits
        print("i=",i)
        print(numberOfCommentsContainWordInClass[i])
        print(numberOfCommentsContainWordInClass[i+1])
        for l in range(len(Karray[i][2])):    # length of the corresponding binary vectors
            for k in range(len(vocabV)):      # 2078 (words)
                if Karray[i][2][l][k] == 1:    # every binary vectors' every word is 1(present)
                    numberOfCommentsContainWordInClass[i][k] += 1   #


    print("``````````````````````````````````")
    print("numberOfCommentsContainWordInClass is ")
    print(numberOfCommentsContainWordInClass[0])
    print(numberOfCommentsContainWordInClass[1])
    #compute the relative frequency of comments of class K
    totalNumberOfComments = N
    priors=[0]*len(Karray)
    for p in range(len(Karray)):
        priors[p]=Karray[p][1]/totalNumberOfComments

    #compute probabilities of each word given the comment class
    likelyhoods=[[0]*len(vocabV)]*len(Karray)
    for q in range(len(Karray)):
        for s in range (len(vocabV)):
            likelyhoods[q][s]=numberOfCommentsContainWordInClass[q][s]/Karray[q][1]

    result = [priors,likelyhoods]
    return result


#To classify an unlabelled comment in C_test,we estimate the posterior probability for each class K
def predict ():
    Keyarray = ["hockey", "nba", "soccer", "baseball", "GlobalOffensive",
         "canada", "conspiracy", "europe", "anime", "Overwatch",
         "wow", "nfl", "leagueoflegends", "trees", "Music",
         "AskReddit", "worldnews", "funny", "gameofthrones", "movies"]
    print("Keyarray(list)", Keyarray)
    print("``````````````````````````````````")
    vocVector=getVocabularyVector()
    print("vocVector(list)", vocVector)
    print("length of vocVector ",len(vocVector))
    vocVector = removeDuplicateWords(vocVector)
    print("--------------remove all duplicates in vocVector-------------")
    print("New vocVector(list)", vocVector)
    print("New length of vocVector ",len(vocVector))
    print("``````````````````````````````````")
    docMatrix=preprocessComments(vocVector)
    print("docMatrix shape", docMatrix.shape)
    print("docMatrix", docMatrix)
    print("``````````````````````````````````")
    priors = fit(vocVector)[0]
    print("``````````````````````````````````")
    print("----------reach?---------------")
    exit()










    print("priors is")
    print(priors)
    likelyhoods = fit(vocVector)[1]
    print("``````````````````````````````````")
    print("likelyhoods is")
    print(likelyhoods)
    for i in range(docMatrix.shape[0]):
        #binaryM = np.zeros((documentM.shape[0],len(vocabV)))
        for j in range(docMatrix.shape[1]):
            for k in range(len(Keyarray)):
                if (labels[i]==Keyarray[k]):
                    if (docMatrix[i][j]==1):
                        docMatrix[i][j]=docMatrix[i][j]*likelyhoods[k][j]
                #likelyhoods=[[0]*len(vocabV)]*len(Karray)
                    else:
                        docMatrix[i][j]=1-likelyhoods[k][j]
    print("``````````````````````````````````")
    print("docMatrix_processed is")
    print(docMatrix)




    product=[1]*docMatrix.shape[0]
    for p in range(docMatrix.shape[0]):
        for q in range(docMatrix.shape[1]):
            if(docMatrix[p][q]!=0):
                product[p]=product[p]+ math.log(docMatrix[p][q])
    print("``````````````````````````````````")
    print("product is")
    print(product)

    #compute posterior probabilities for each comment based on 20 classes
    #the final prediction will be the max of all the posterior probabilities
    posProb = [[0]*len(priors)]*len(product)
    predictionArray=[0]*len(product)
    for x in range(len(product)):
        for y in range(len(priors)):
            posProb[x][y]=product[x]*priors[y]
    print("``````````````````````````````````")
    print("posProb is")
    print(posProb)
    for z in range(len(posProb)):
        maximumProb = max(posProb[z])
        indexOfMaxima=posProb[z].index(maximumProb)
        predictionArray[z]=Keyarray[indexOfMaxima]

    return predictionArray

print(*predict(),sep=",")