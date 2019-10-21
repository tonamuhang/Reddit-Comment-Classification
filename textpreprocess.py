import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
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
from bs4 import BeautifulSoup
import urllib
from urllib.request import urlopen
from urlextract import URLExtract

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

class RepeatReplacer():
    def __init__(self):
        self.repeat_reg = re.compile(r'(\w*)(\w)\2(\w*)')
        self.repl = r'\1\2\3'
    def replace(self, word):
        # check if it is a standard word
        if wordnet.synsets(word):
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


def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english')).union(["gt"])
    extractor = URLExtract()
    info = " "

    for url in extractor.gen_urls(text):
        try:
            if "youtube" in url or "youtu.be" in url:
                content = urlopen(url, timeout=1).read()
                content = BeautifulSoup(content).find('title').string  # HTML decoding
                info += " " + content
        except:
            continue

    text += info
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub(' ', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = remove_special_chars_digits(text)
    text = word_tokenize(text)
    text = remove_repeat(text)
    text = np.asarray(text)
    text = lemmatize_new(text)

    text = ' '.join(word for word in text if word not in STOPWORDS)

    TextPreprocess.num += 1
    print(TextPreprocess.num, " " + text)
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

def lemmatize_new(text):
    lemmatizer = nltk.WordNetLemmatizer()
    for i, w in enumerate(text):
        w = lemmatizer.lemmatize(w, 'v')
        text[i] = lemmatizer.lemmatize(w)

    return text

def remove_repeat(text):
    replacer = RepeatReplacer()
    for i, w in enumerate(text):
        w = replacer.replace(w)
        text[i] = w
    return text

class TextPreprocess:

    num = 0

    @staticmethod
    # Remove tags
    def remove_tags(text):
        text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
        return text

    @staticmethod
    def remove_repeat(text):
        replacer = RepeatReplacer()
        for row_i in range(text.shape[0]):
            for word_j in range(len(text[row_i])):
                text[row_i][word_j] = replacer.replace(text[row_i][word_j])
        return text

    @staticmethod
    # Input: File location train(str)
    # Output: Preprocessed file(dataframe)
    def process(X):
        # Read in files
        train = pd.read_csv(X, sep=',')
        #
        # remove_tags = np.vectorize(TextPreprocess.remove_tags)
        # remove_repeat = np.vectorize(TextPreprocess.remove_repeat)
        #
        # # Note: Order of remove operations matters!
        # comments = comments.apply(lambda x: remove_tags(x))
        # comments = comments.apply(lambda x: remove_special_chars_digits(x))
        # print("----removed urls tags, set lowercase, remove specials characters and digits----\n", comments.head(5))
        #
        # # Tokenize comments
        # comments = comments.apply(word_tokenize)
        # print("----tokenized----\n", comments.head(12))
        #
        # comments = lemmatize(comments)
        # print("----lemmatized----\n", comments.head(12))
        # #
        # # # Remove Stopwords
        # # stop_words = set(stopwords.words('english'))
        # # comments = comments.apply(lambda x: [item for item in x if item not in stop_words])
        # # print("----removed stop words----\n", comments.head(5))
        # #
        #
        # comments = remove_repeat(comments)
        # print("----removed repeat----\n", comments.head(5))

        train['comments'] = train['comments'].apply(clean_text)
        comments = train['comments']

# Uncomment to choose between train and test processed
        #comments.to_csv('train_processed.csv')
        comments.to_csv('test_processed.csv')

        return comments

# Note:
# This program will output a csv file named train_processed.csv
#    which is the result of reddit_traint.csv being preprocessed.

# How to use
# Uncomment to choose between train and test processed
#TextPreprocess.process("reddit_train.csv")
TextPreprocess.process("reddit_test.csv")


