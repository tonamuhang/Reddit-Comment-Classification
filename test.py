import pandas as pd
import numpy as np
from sklearn import feature_extraction, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
# load data
from sklearn.pipeline import Pipeline

train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']

comment = train['comments']
labels_list = train['subreddits']

labels = labels_list

# Process Data
X_train, X_test, y_train, y_test = train_test_split(comment, labels, random_state = 0)
cvec = CountVectorizer(min_df= 1, stop_words='english')
X_train_counts = cvec.fit_transform(X_train)
X_test_counts = cvec.transform(X_test)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# # TODO: Fix y format
# # Start to fit
# print("Starting to fit...")
# clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
# clf.fit(X_train_tfidf, y_train)
#
# # Predict
# print("Starting to predict")
# predicted = clf.predict(X_test_tfidf)
# print(accuracy_score(y_test, predicted))

# Pipline format, follow this:
pipeline = Pipeline([('vect', feature_extraction.text.CountVectorizer(ngram_range=(1, 1), stop_words='english')),
                     ('tfidf', feature_extraction.text.TfidfTransformer()),
                     ('norm', preprocessing.Normalizer()),
                     ('clf', LogisticRegression(multi_class='multinomial', solver='newton-cg'))])


parameters = {'clf__C': [1, 2, 5, 10]}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)

print("Starting to fit")
grid.fit(X_train, y_train)

print ("score = %3.2f" %(grid.score(X_test, y_test)))
print (grid.best_params_)