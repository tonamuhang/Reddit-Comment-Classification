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
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class RepeatReplacer():
    def __init__(self):
        self.repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def replace(self, word):
        if wordnet.synsets(word):  # 判断当前字符串是否是单词
            return word
        repl_word = self.repeat_reg.sub(self.repl, word)
        if repl_word != word:
            return self.replace(repl_word)
        else:
            return repl_word


# Remove URLs
def remove_URLs(text):
    text = re.sub(r"http\S+", " ", text)
    return text


# Set all words to lowercase
def set_lowercase(text):
    text = text.lower();
    return text


# Remove tags
def remove_tags(text):
    text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    return text


# Remove special characters and digits
def remove_special_chars_digits(text):
    text = re.sub("(\\d|\\W)+", " ", text)
    return text

def lemmatize(text):
    lemmatizer = nltk.WordNetLemmatizer()
    for row_i in range(text.shape[0]):
        for word_j in range(len(text[row_i])):
            text[row_i][word_j] = lemmatizer.lemmatize(text[row_i][word_j])
    return text

def remove_repeat(text):
    replacer = RepeatReplacer()
    for row_i in range(text.shape[0]):
        for word_j in range(len(text[row_i])):
            text[row_i][word_j] = replacer.replace(text[row_i][word_j])
    return text


class TextPreprocess:
    @staticmethod
    # Input: File location train(str)
    # Output: Preprocessed file(dataframe)
    def process(X):
        # Read in files
        train = pd.read_csv(X, sep=',')
        test = pd.read_csv("reddit_test.csv", sep=',')

        comments = train['comments']
        test_x = test['comments']
        labels = train['subreddits']

        # Note: Order of remove operations matters!
        comments = comments.apply(lambda x: remove_URLs(x))
        comments = comments.apply(lambda x: set_lowercase(x))
        comments = comments.apply(lambda x: remove_tags(x))
        comments = comments.apply(lambda x: remove_special_chars_digits(x))
        print("----remove urls tags, set lowercase, remove specials characters and digits----\n", comments.head(5))

        # Tokenize comments
        comments = comments.apply(word_tokenize)
        print("----tokenized----\n", comments.head(12))

        # comments = lemmatize(comments)
        # print("----lemmatized----\n", comments.head(12))
        #
        # # Remove Stopwords
        # stop_words = set(stopwords.words('english'))
        # comments = comments.apply(lambda x: [item for item in x if item not in stop_words])
        # print("----removed stop words----\n", comments.head(5))
        #
        # comments = remove_repeat(comments)
        # print("----removed repeat----\n", comments.head(5))

        return comments


# How to use
# X = TextPreprocess.process("reddit_train.csv")



