import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.svm import SVC

print(os.listdir("./input"))

data = pd.read_json("./input/Sarcasm_Headlines_Dataset.json", lines = True)
data = data[["headline", "is_sarcastic"]]

vectorizer = TfidfVectorizer()

X_train, X_test, y_train, y_test = train_test_split(data['headline'].values, data['is_sarcastic'].values, test_size=0.15, random_state=101)

# X_train_raw = data['headline'].values

X_train = vectorizer.fit_transform(X_train)
# Y_train = data['is_sarcastic'].values

X = X_train.toarray()

model = SVC()
model.fit(X, y_train)

X_test = vectorizer.transform(X_test)
y_pred = model.predict(X_test)

print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print("F1 Score: ", metrics.f1_score(y_test, y_pred))


# pca = PCA(n_components=2)
# pca.fit(X)
# X_new = pca.transform(X)
# plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=Y_train)
# plt.legend()
# plt.savefig('PCA.png')
# plt.show()