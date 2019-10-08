import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score

class BNB:

    def __init__(self):
        # Probabilities
        self.likelihood = {}
        self.prob_matrix = {}

        # Classes
        self.classes = None

    # Input: None
    # Output: None
    # Usage: Create a dictionary
    def create_dictionary(self, sentence):
        sentence_dict = {}
        prev = ''
        for word in sentence.split():
            if word not in sentence_dict:
                sentence_dict[word] = []
            sentence_dict[word].append(prev)
            prev = word

        return sentence_dict

    # Input: Dataframe m*n, dataframe n*1
    # Output: None
    def fit(self, feature, label, alpha=1):
        # Load data
        X = feature
        y = label

        # Dictionaries for features and classes
        self.classes = y.to_frame().subreddits.unique()
        num_feature = len(X)
        print(num_feature)
        num_class = len(self.classes)

        for k in self.classes:
            # Nk be the total number of documents of that class
            Nk = len(y[y == k])
            # nkwt be the number of documents of class k in which wt is observed
            nkwt = len(X[y == k])

            # Calculate P(y) with laplace smoothing
            self.likelihood[k] = (alpha + Nk) / (num_class * alpha + num_feature)

            # Calculate P(X|y) with laplace smoothing
            self.prob_matrix[k] = (alpha + nkwt) / (num_feature * alpha + Nk * num_feature)

        print("Fit Finished")
        return self

    # Predict on a single class.
    def predict_helper(self, sentence):
        result = None
        features = self.create_dictionary(sentence)
        max_prob = -999999999

        for k in self.classes:
            prior = self.prob_matrix[k]


            # TODO: Calculate posterior
            # posterior = [bt P(wt |Ck) + (1−bt) (1−P(wt |Ck))]
            # P(wt |Ck) = self.likelihood[k]
            # bt = ???
            
            # Probability = prior * posterior
            # if the new prob is greater than the current max, replace it and the predicted result
            prob = prior
            if prob > max_prob:
                max_prob = prob
                result = k

        return result

    def predict(self, X):
        result = {}
        for sentence in X:
            result[sentence] = self.predict_helper(sentence)

        return result

    def score(self, X, y):
        predicted = self.predict(X)
        # return accuracy_score(y, predicted)


# Read in files
train = pd.read_csv("reddit_train.csv", sep=',')
comments = train['comments']
labels = train['subreddits']

# Process Data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(comments, labels, random_state = 0, test_size=0.5)

bnb = BNB()
bnb.fit(X_train, y_train)

print(bnb.score(X_test, y_test))

