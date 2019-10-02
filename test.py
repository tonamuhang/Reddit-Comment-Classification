import pandas as pd
import numpy as np
from sklearn import feature_extraction
import matplotlib.pyplot as plt
import matplotlib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# load data
train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']
# test_y = test['subreddits']
train_x = train['comments']
train_y = train['subreddits']


# fig = plt.figure(figsize=(8,6))
# plt.tight_layout()
# train.groupby('subreddits').subreddits.count().plot.bar(ylim=0)
# plt.show()

# pipline = Pipeline([('clf', LogisticRegression(C=10, multi_class='multinomial')),
#                     ('vect', feature_extraction.text.CountVectorizer(ngram_range=(1, 2), stop_words='english'),
#                      ('tfidf', feature_extraction.text.TfidfTransformer()))])
#
# pipline.fit(train_x, train_y)

# Process data
print("Processing data")
cvec = feature_extraction.text.CountVectorizer(ngram_range=(1, 1), stop_words='english')
train_x = cvec.fit(train_x).transform(train_x)

# Start to fit
print("Starting to fit...")
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(train_x, train_y)

# Predict
print("Starting to predict")
predicted = clf.predict(test_x)
print(predicted)