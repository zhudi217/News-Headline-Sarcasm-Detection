import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold

print(os.listdir("./input"))

with open('./input/Sarcasm_Headlines_Dataset.json', 'r') as f:
    jsonDict = f.readlines()

print(jsonDict[:5])
print(len(jsonDict))

# remove the trailing "\n" from each line, I did not do any real cleaning I wanted to see the pure accurasy, more in the few next cells
data = list(map(lambda x: x.rstrip(), jsonDict))
print(data[:5])

data_json_str = "[" + ",".join(data) + "]"

print(len(data_json_str))
print(data_json_str[:10])

# now, load it into pandas
data_df = pd.read_json(data_json_str)
print(data_df.head(10))

### Task #2: Check for missing values:
data_df.isnull().sum()

# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i, link, headline, sarcastic in data_df.itertuples():  # iterate over the DataFrame
    if type(headline) == str:  # avoid NaN values
        if headline.isspace():  # test 'review' for whitespace
            blanks.append(i)  # add matching index numbers to the list


print(list(data_df.itertuples())[:5])
print(len(blanks))
print(data_df.shape)

### Task #3:  Remove NaN values:
data_df.dropna(inplace=True)
# print(data_df.shape)
headlines = data_df['headline']
labels = data_df['is_sarcastic']

# print(headlines[0], labels[0])

#Sanity check
data_df['is_sarcastic'].value_counts()


### Task #5: Split the data into train & test sets: use a validation at the end of phd to twiddle a little more
#print(len(X))
X = data_df['headline']#.append(df['headline'])
#X.append(data_df['headline'])
y = data_df['is_sarcastic']#.append(df['is_sarcastic'])
# print(X[:10], len(X))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
# print(X_test.shape)

### Task #6: Build a pipeline to vectorize the date, then train and fit a model
vectorizer = TfidfVectorizer()
# print(X_train)
X_train_vec = vectorizer.fit_transform(X_train, y_train)
X_test_vec = vectorizer.transform(X_test, y_test)

# text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
model = LinearSVC()
model.fit(X_train_vec, y_train)

### Task #7: Run predictions and analyze the results
# Form a prediction set
pred = model.predict(X_test_vec)


# Report the confusion matrix
print(confusion_matrix(y_test, pred))

# Print a classification report
print(classification_report(y_test, pred, digits=4))