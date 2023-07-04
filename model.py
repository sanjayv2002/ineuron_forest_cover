# Importing packages

import pandas as pd 
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  accuracy_score


# Importing data

data = pd.read_csv("data/train.csv")
X = data.drop(['Cover_Type'], axis = 1)
Y = data['Cover_Type']

#  Splitting train and test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Normalizing

X_train = X_train.drop(['Id'], axis = 1)
X_test = X_test.drop(['Id'], axis = 1)

norm = Normalizer()
norm_x_train = norm.fit_transform(X_train)
norm_x_test = norm.transform(X_test)

tree = DecisionTreeClassifier()

tree.fit(norm_x_train, Y_train)

pred = tree.predict(norm_x_test)

acc = accuracy_score(Y_test, pred)

print(acc)