import pandas as pd
import numpy as np
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
import re
from sklearn.preprocessing import StandardScaler
import nltk
tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)

# Modify original stop words
my_stopwords = text.ENGLISH_STOP_WORDS.union(["january", "february", "march", "april", "may", "june",
                                              "july", "august", "september", "october", "november",
                                              "december", "tomorrow", "today", "yesterday", "hundred", "thousand",
                                              "million"])

# Load data
from sklearn.pipeline import Pipeline

train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']

comment = train['comments']
# for i, c in enumerate(comment):
#         comment.set_value(i, re.sub(r"http\S+", "", c))


labels_list = train['subreddits']

labels = labels_list

# Process Data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(comment, labels, random_state = 1)

# Pipline format, follow this:
pipeline_LogisticRegression = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', LogisticRegression(multi_class='multinomial', solver='newton-cg'))
                     ])

#-----------------------------------------------------------------------
# # Support Vector Machine
pipeline_LinearSVC = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(sublinear_tf=True,norm='l1')),
                     ('norm', Normalizer()),
                     ('clf', LinearSVC(C=0.215,max_iter=5000,penalty='l2'))
                    ])

#-----------------------------------------------------------------------
# Decision Tree
pipeline_DecisionTree = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', tree.DecisionTreeClassifier())
                    ])
#-----------------------------------------------------------------------
# Bernoulli Naive Bayes
pipeline_BernoulliNaiveBayes = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', BernoulliNB(alpha=1.0))
                    ])

#-------------------------------------------------------------
parameters_LogisticRegression = {'clf__C': [1, 2, 5, 10]} # C: inverse of regularization strength

parameters_LinearSVC = {}  # ignore best parameters search

parameters_DecisionTree = {'clf__max_depth': [1024, 2048, 5096]} # max_depth: Not too high(over fitting) nor too low(under fitting)
parameters_BernoulliNaiveBayes = {'clf__alpha':[0.5, 1.0, 2.0, 3.0]} # alpha: Additive (LaPlace/Lidstone) smoothing paramter

grid_LogisticRegression = GridSearchCV(pipeline_LogisticRegression, param_grid=parameters_LogisticRegression, cv=5)
grid_LinearSVC = GridSearchCV(pipeline_LinearSVC, param_grid=parameters_LinearSVC, cv=5, n_jobs=-1, verbose=3)
grid_DecisionTree = GridSearchCV(pipeline_DecisionTree, param_grid=parameters_DecisionTree, cv=5, verbose=3, n_jobs=5)
grid_BernoulliNaiveBayes = GridSearchCV(pipeline_BernoulliNaiveBayes, param_grid=parameters_BernoulliNaiveBayes, cv=5)

# print("----------------------------------------------------------------------")   # 0.54   clf__c: 2
# print("Logistic Regression")
# grid_LogisticRegression.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_LogisticRegression.score(X_test, y_test)))
# print (grid_LogisticRegression.best_params_)
#

print("----------------------------------------------------------------------")
# print("Linear SVC")
# grid_LinearSVC.fit(X_train, y_train)
# print ("score = %3.4f" %(grid_LinearSVC.score(X_test, y_test)))
# print (grid_LinearSVC.best_params_)


# print("----------------------------------------------------------------------")        # 0.32 max_depth: 2048
# print("Decision Tree")
# grid_DecisionTree.fit(X_train, y_train)
# print ("score = %3.3f" %(grid_DecisionTree.score(X_test, y_test)))
# print (grid_DecisionTree.best_params_)

# print("----------------------------------------------------------------------")     # 0.40 clf__alpha: 0.5
# print("Bernoulli Naive Bayes")
# grid_BernoulliNaiveBayes.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_BernoulliNaiveBayes.score(X_test, y_test)))
# print (grid_BernoulliNaiveBayes.best_params_)

# Todo: Generate id column
# prediction = grid_LinearSVC.predict(test_x)
#
# prediction = pd.DataFrame(prediction, columns=['Category']).to_csv('prediction.csv')

# ------------------------------------------------------------------------------------------------
# Directly fit the whole train set
print("Linear SVC")
grid_LinearSVC.fit(comment, labels_list)
prediction = grid_LinearSVC.predict(test_x)
prediction = pd.DataFrame(prediction, columns=['Category']).to_csv('prediction.csv')
