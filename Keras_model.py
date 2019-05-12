import json
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from collections import Counter
import tensorflow as tf
import keras
from keras import models
from keras import layers
import numpy as np
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, SGD, RMSprop

# Just disables the warning, doesn't enable AVX/FMA
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Convert data to a certain dimension
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


data = []
for line in open('./input/Sarcasm_Headlines_Dataset.json', 'r'):
    data.append(json.loads(line))
titles = []
y_vals = []


for i in range(0,len(data)):
    titles.append(data[i]['headline'])
    y_vals.append(data[i]['is_sarcastic'])


titles_tokenized = []
for title in titles:
    titles_tokenized.append(word_tokenize(title))
titles_an = [] # alphanumeric
for title in titles_tokenized:
    words = [word for word in title if word.isalpha()]
    titles_an.append(words)
# Let's now stem the words
porter = PorterStemmer()
titles_preprocessed = []
for title in titles_an:
    stemmed = [porter.stem(word) for word in title]
    titles_preprocessed.append(stemmed)
# Now, let's create a large list of all of the words and find the 10,000 most frequent ones
word_list = []

for title in titles_preprocessed:
    for word in title:
        word_list.append(word)

freq_list = Counter(word_list)
dictionary = freq_list.most_common(15000)
dictionary = list(zip(*dictionary))[0]

# We now have a list with the 10000 most common words. Let us convert our sentences to lists of these words in
# order to feed it into the Neural Network.
nums = range(0, 10000)
word_int = dict(zip(dictionary, nums))
x_vals = []

for title in titles_preprocessed:
    x_vals.append([word_int[x] for x in title if x in word_int.keys()])


X = np.array(x_vals)
y = np.asarray(y_vals).astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

X_train = vectorize_sequences(X_train)
X_test = vectorize_sequences(X_test)


# X_train = np.reshape(X_train, (1, X_train.shape[0], X_train.shape[1]))
# X_test = np.reshape(X_test, (1, X_test.shape[0], X_test.shape[1]))
#
# y_train = np.reshape(y_train, (1, y_train.shape[0], 1))
# y_test = np.reshape(y_test, (1, y_test.shape[0], 1))

# print(X_train.shape)
# print(y_train.shape)


# Prevent Tensorflow from allocating my entire GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


# Let us train the model with 6 epochs.
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(8, activation='relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))

# model.add(LSTM(16, return_sequences=True, input_shape = (24038, 10000), activation='relu'))
# model.add(LSTM(4, return_sequences=True, activation='relu'))
# model.add(LSTM(1, return_sequences=True, name='output', activation='sigmoid'))

model.compile(optimizer = 'adagrad', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 3, batch_size = 512)

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

y_pred = model.predict(X_test) > 0.51

print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("F1 Score: ", f1_score(y_test, y_pred))