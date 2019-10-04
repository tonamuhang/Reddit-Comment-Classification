import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

# load data
train = pd.read_csv("reddit_train.csv", sep=',')
test = pd.read_csv("reddit_test.csv", sep=',')
test_x = test['comments']

comment = train['comments']
labels_list = train['subreddits']
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
features = tfidf.fit_transform(comment)
labels = labels_list

# fig = plt.figure(figsize=(8,6))
# plt.tight_layout()
# train.groupby('subreddits').subreddits.count().plot.bar(ylim=0)
# plt.show()

# Pipline format, follow this:
# pipline = Pipeline([('clf', LogisticRegression(C=10, multi_class='multinomial')),
#                     ('vect', feature_extraction.text.CountVectorizer(ngram_range=(1, 2), stop_words='english'),
#                      ('tfidf', feature_extraction.text.TfidfTransformer()))])
#
# pipline.fit(train_x, train_y)

# Process Data
X_train, X_test, y_train, y_test = train_test_split(comment, labels, random_state = 0)
cvec = CountVectorizer()
X_train_counts = cvec.fit_transform(X_train)
X_test_counts = cvec.transform(X_test)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# TODO: Fix y format
# Start to fit
print("Starting to fit...")
clf = LogisticRegression(random_state=0, multi_class='multinomial', solver='newton-cg')
clf.fit(X_train_tfidf, y_train)

# Predict
print("Starting to predict")
predicted = clf.predict(X_test_tfidf)
print(predicted)