from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv

train_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.train.csv')
train_x_tfidf = train_tfidf.iloc[:, 1:]
train_y_tfidf = train_tfidf.iloc[:, 0]

test_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.test.csv')
test_x_tfidf = test_tfidf.iloc[:, 1:]
test_y_tfidf = test_tfidf.iloc[:, 0]

eval_df_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.eval.anon.csv')
eval_df_tfidf = eval_df_tfidf.iloc[:, 1:]

print("Data Loaded")

train_y_tfidf = train_y_tfidf.replace(0, -1)
test_y_tfidf = test_y_tfidf.replace(0, -1)

print("Data Changed")

clf1 = LogisticRegression(max_iter=1250, random_state=65)
clf1.fit(train_x_tfidf, train_y_tfidf)


pred = clf1.predict(train_x_tfidf)

accuracy = accuracy_score(train_y_tfidf, pred)
print("Voting classifier training accuracy:", accuracy)

pred = clf1.predict(test_x_tfidf)


accuracy = accuracy_score(test_y_tfidf, pred)
print("Voting classifier test accuracy:", accuracy)

eval_pred = clf1.predict(eval_df_tfidf)



file = open('logistic_regression.csv', 'w', newline='')

with file:
    header = ['example_id', 'label']
    writer = csv.DictWriter(file, fieldnames=header)

    writer.writeheader()
    for i in range(len(eval_pred)):
        if eval_pred[i] == -1:
            val = 0
        else:
            val = 1
        writer.writerow({'example_id': i, 'label': val})

