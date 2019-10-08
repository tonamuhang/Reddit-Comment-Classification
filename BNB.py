import pandas as pd
import numpy as np
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
from sklearn.metrics import accuracy_score
import textpreprocess
from sklearn.feature_extraction import DictVectorizer


class BNB:

    def __init__(self):
        # Probabilities
        self.likelihood = {}
        self.prob_matrix = {}

        # Classes
        self.classes = None

        # Vocabulary of the whole document
        self.V = None

        # Vocabulary of each class
        self.Vc = {}

    # Input: Dataframe
    # Output: None
    def create_feature(self, X, min_df=5):
        # X = textpreprocess.remove_consecutive(X)

        v = CountVectorizer(stop_words='english', min_df=min_df, )
        X = v.fit_transform(X)

        # print(v.get_feature_names())
        return v, X

    # Input: Dataframe m*n, dataframe n*1
    # Output: None
    def fit(self, feature, label):
        # Load data
        X = feature
        y = label

        # Dictionaries for features and classes
        self.classes = y.to_frame().subreddits.unique()
        self.V, bag_of_words = self.create_feature(X)
        num_sample = len(X)
        num_class = len(self.classes)

        # Counts
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in self.V.vocabulary_.items()]

        for k in self.classes:
            # Nk be the total number of documents of that class
            Nk = len(y[y == k])
            self.prob_matrix[k] = {}

            # Build dictionary for the class
            self.Vc[k] = self.create_feature(X[y == k])
            print(k, " ######################################")
            # print(self.Vc[k])

            # # nkwt be the number of documents of class k in which wt is observed
            for w in self.Vc[k][0].vocabulary_:
                nkwt = 0
                for t in words_freq:
                    if t[0] == w:
                        nkwt += t[1]
                        # Calculate P(X|y) with + 1 laplace smoothing
                        self.prob_matrix[k][w] = (1 + nkwt) / (2 + Nk)
                        # print((1 + nkwt) / (2 + Nk))

            # Calculate P(y)
            self.likelihood[k] = Nk / num_sample

        print("Fit Finished")
        return self

    # Predict on a single class.
    def predict_helper(self, sentence):
        result = None
        max_prob = -999999999
        features, count = self.create_feature(sentence, min_df=1)

        for k in self.classes:
            prior = self.likelihood[k]

            # TODO: Calculate posterior
            # posterior = [bt P(wt |Ck) + (1−bt) (1−P(wt |Ck))]
            # P(wt |Ck) = self.likelihood[k]
            # bt = ???
            posterior = 1
            for f in self.V.vocabulary_:
                if f in features.vocabulary_:
                    if f in self.Vc[k]:
                        posterior *= self.prob_matrix[k][f]

                    else:
                        posterior *= (1 - self.prob_matrix[k][f])


            # Probability = prior * posterior
            # if the new prob is greater than the current max, replace it and the predicted result
            prob = prior * posterior

            if prob > max_prob:
                max_prob = prob
                result = k

        return result

    def predict(self, X):
        result = {}
        i = 0.0
        total = len(X)
        for sentence in X:
            i += 1
            print("Completed: ", i/total)
            result[sentence] = self.predict_helper([sentence])
            print(sentence)
            print(result[sentence])

        return result

    def score(self, X, y):
        predicted = self.predict(X)
        return accuracy_score(y, predicted)


# Read in files
train = pd.read_csv("reddit_train.csv", sep=',')
comments = train['comments']
labels = train['subreddits']

# Process Data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(comments, labels, random_state=0, test_size=0.2)

bnb = BNB()
bnb.fit(X_train, y_train)
bnb.score(X_test, y_test)
