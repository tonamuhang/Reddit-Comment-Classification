import numpy as np
import math

class BNB:

    def __init__(self, classes):
        self.dictionary = set([])

    # Input: None
    # Output: None
    # Usage: Create a dictionary
    def create_dictionary(self):
        wordset = set([])
        for sentence in self.X:
            wordset = wordset.union(set(sentence))
        self.dictionary = wordset

    # Input: Dataframe m*n, dataframe n*1
    # Output: None
    def fit(self, feature, label, alpha=1):
        # Load data
        X = feature
        y = np.array(label)

        # Dictionaries for features and classes
        num_feature = {}
        num_class = {}

        # Probablities
        prob_class = {}

        for i, c in enumerate(label):
            # Record the class if it hasnt been seen before
            num_class[c] = num_class.get(c, 0) + 1

            # Record
            for w in X[i].keys():
                if w in num_feature:
                    num_feature[w][c] = num_feature[w].get(c, 0) + 1
                else:
                    num_feature[w] = {c: 1}