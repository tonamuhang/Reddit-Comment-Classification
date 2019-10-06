import pandas as pd
import numpy as np
from sklearn import feature_extraction, preprocessing, tree
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Load data
from sklearn.pipeline import Pipeline

train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']

comment = train['comments']
labels_list = train['subreddits']

labels = labels_list

# Process Data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(comment, labels, random_state = 0)

# Pipline format, follow this:
pipeline_LogisticRegression = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', LogisticRegression(multi_class='multinomial', solver='newton-cg'))
                     ])

#-----------------------------------------------------------------------
# # Support Vector Classification
pipeline_SVC = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', SVC(gamma='scale'))
                    ])
#-----------------------------------------------------------------------
# # Support Vector Machine
pipeline_LinearSVC = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', LinearSVC())
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
parameters_SVCandLinearSVC = {'clf__C': [1.0, 2.0, 5.0, 10.0]} # C: penalty parameter C of the error term
parameters_DecisionTree = {} # min_samples_split: MIN number of samples required to split an internal node
parameters_BernoulliNaiveBayes = {'clf__alpha':[0.5, 1.0, 2.0, 3.0]} # alpha: Additive (LaPlace/Lidstone) smoothing paramter

grid_LogisticRegression = GridSearchCV(pipeline_LogisticRegression, param_grid=parameters_LogisticRegression, cv=5)
grid_SVC = GridSearchCV(pipeline_SVC, param_grid=parameters_SVCandLinearSVC, cv=5)
grid_LinearSVC = GridSearchCV(pipeline_LinearSVC, param_grid=parameters_SVCandLinearSVC, cv=5)
grid_DecisionTree = GridSearchCV(pipeline_DecisionTree, param_grid=parameters_DecisionTree, cv=5, verbose=3, n_jobs=5)
grid_BernoulliNaiveBayes = GridSearchCV(pipeline_BernoulliNaiveBayes, param_grid=parameters_BernoulliNaiveBayes, cv=5)

# print("----------------------------------------------------------------------")   # 0.54   clf__c: 2
# print("Logistic Regression")
# grid_LogisticRegression.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_LogisticRegression.score(X_test, y_test)))
# print (grid_LogisticRegression.best_params_)

# print("----------------------------------------------------------------------")
# print("SVC")
# grid_SVC.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_SVC.score(X_test, y_test)))
# print (grid_SVC.best_params_)

# print("----------------------------------------------------------------------")      # 0.56  clf__c: 1.0
# print("Linear SVC")
# grid_LinearSVC.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_LinearSVC.score(X_test, y_test)))
# print (grid_LinearSVC.best_params_)

print("----------------------------------------------------------------------")
print("Decision Tree")
grid_DecisionTree.fit(X_train, y_train)
print ("score = %3.3f" %(grid_DecisionTree.score(X_test, y_test)))
print (grid_DecisionTree.best_params_)

# print("----------------------------------------------------------------------")     # 0.40 clf__alpha: 0.5
# print("Bernoulli Naive Bayes")
# grid_BernoulliNaiveBayes.fit(X_train, y_train)
# print ("score = %3.2f" %(grid_BernoulliNaiveBayes.score(X_test, y_test)))
# print (grid_BernoulliNaiveBayes.best_params_)