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
                     ('clf', SVC())
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
pipeline_DecisionTree = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', tree.DecisionTreeClassifier())
                    ])
#-----------------------------------------------------------------------
# Bernoullie Naive Bayes
pipeline_BernoullieNaivesBayes = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), stop_words='english')),
                     ('tfidf', TfidfTransformer()),
                     ('norm', Normalizer()),
                     ('clf', BernoulliNB(alpha=1.0))
                    ])

#-------------------------------------------------------------
parameters = {'clf__C': [1, 2, 5, 10]}

#grid = GridSearchCV(pipeline_LogisticRegression, param_grid=parameters, cv=5)
#grid = GridSearchCV(pipeline_SVC, param_grid=parameters, cv=5)
#grid = GridSearchCV(pipeline_LinearSVC, param_grid=parameters, cv=5)
#grid = GridSearchCV(pipeline_DecisionTree, param_grid=parameters, cv=5)
#grid = GridSearchCV(pipeline_BernoullieNaivesBayes, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
print ("score = %3.2f" %(grid.score(X_test, y_test)))
print (grid.best_params_)
