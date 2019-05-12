import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, Embedding, Dropout, Conv1D
from keras.layers import GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC
from sklearn.model_selection import KFold

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

print(os.listdir("./input"))

data = pd.read_json("./input/Sarcasm_Headlines_Dataset.json", lines = True)
data = data[["headline", "is_sarcastic"]]
print(data.shape)
data.head(10)

analyzer = SentimentIntensityAnalyzer()

final_list = []
for sent in data['headline']:
    senti = analyzer.polarity_scores(sent)
    list_temp = []
    for key, value in senti.items():
        temp = value
        list_temp.append(temp)
    final_list.append(list_temp)


temp_df = pd.DataFrame(final_list, columns=['compound','neg','neu','pos'], index = data.index)
data = pd.merge(data, temp_df, left_index=True,right_index=True)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data.head(5))


train_df, test_df = train_test_split(data, test_size=0.15, random_state=101)
train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=101)
print("Train size:{}".format(train_df.shape))
print("Validation size:{}".format(val_df.shape))
print("Test size:{}".format(test_df.shape))


embed_size = 300
max_features = 50000
maxlen = 100


## fill up the missing values
train_X = train_df["headline"].fillna("_na_").values
val_X = val_df["headline"].fillna("_na_").values
test_X = test_df["headline"].fillna("_na_").values


## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)


## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)


## Get the target values
train_y = train_df['is_sarcastic'].values
val_y = val_df['is_sarcastic'].values
test_y = test_df['is_sarcastic'].values


''' CNN '''
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Conv1D(256, maxlen)(x)
x = GlobalMaxPool1D()(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1, activation="sigmoid")(x)
model2 = Model(inputs=inp, outputs=x)
adam =  Adam(lr=0.0001,decay=0.00001)
model2.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

print(model2.summary())

model2.fit(train_X, train_y, batch_size=512, epochs=15, validation_data=(val_X, val_y))


# model2.save('cnn.h5')
# model2.save_weights('cnn_weights.h5')

y_pred2 = model2.predict([test_X], batch_size=512, verbose=1)
y_pred2 = y_pred2 > 0.4


print("Accuracy Score: ", accuracy_score(test_y, y_pred2))
print("Confusion Matrix: \n", confusion_matrix(test_y, y_pred2))
print("F1 Score: ", metrics.f1_score(test_y, y_pred2))


''' Appending data from CNN '''
data_X = data["headline"].fillna("_na_").values
data_X = tokenizer.texts_to_sequences(data_X)
data_X = pad_sequences(data_X, maxlen = maxlen)

# y_pred_data1 = model1.predict([data_X], batch_size=512, verbose=1)
y_pred_data2 = model2.predict([data_X], batch_size=512, verbose=1)

d2 = pd.DataFrame(y_pred_data2,columns=['CNN'], index=data.index)

data = pd.merge(data, d2, left_index=True, right_index=True)

data.head()

temp_X = data[['compound', 'neg', 'neu', 'pos', 'CNN']]
temp_y = data['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, test_size=0.33, random_state=101)


''' Trying kFold with Support Vector Classifier '''
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)

# cv = KFold(n_splits=10, random_state=42, shuffle=True)
# scores = []
# i = 1
# for train_index, test_index in cv.split(temp_X):
#     X_train, X_test = temp_X.values[train_index], temp_X.values[test_index]
#     y_train, y_test = temp_y[train_index], temp_y[test_index]
#
#     svc.fit(X_train, y_train)
#     print("Iteration: ", i, " - Score = ", svc.score(X_test, y_test).round(3))
#     scores.append(svc.score(X_test, y_test))
#     i += 1

y_pred_svc = svc.predict(X_test)

print("Accuracy Score: ", accuracy_score(y_test, y_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred_svc))
print("F1 Score: ", metrics.f1_score(y_test, y_pred_svc))
