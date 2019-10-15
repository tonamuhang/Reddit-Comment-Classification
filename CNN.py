import pandas as pd
import numpy as np
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers import Flatten
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

train = pd.read_csv("reddit_train.csv", sep=',')
labels = train['subreddits']

own_embedding_vocab_size = 60000
encoded_docs_oe = [one_hot(d, own_embedding_vocab_size) for d in train['comments']]
print(encoded_docs_oe)

maxlen = 300
padded_docs_oe = pad_sequences(encoded_docs_oe, maxlen=maxlen, padding='post')
print(padded_docs_oe)

model = Sequential()
model.add(Embedding(input_dim=own_embedding_vocab_size, # 10
                    output_dim=32,
                    input_length=maxlen))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])  # Compile the model
print(model.summary())  # Summarize the model
model.fit(padded_docs_oe, labels, epochs=50, verbose=0)  # Fit the model
loss, accuracy = model.evaluate(padded_docs_oe, labels, verbose=0)  # Evaluate the model
print('Accuracy: %0.3f' % accuracy)