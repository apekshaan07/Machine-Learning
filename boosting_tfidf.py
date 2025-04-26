import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import csv
from math import exp



train_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.train.csv')
train_x_tfidf = train_tfidf.iloc[:, 1:]
train_y_tfidf = train_tfidf.iloc[:, 0]

test_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.test.csv')
test_x_tfidf = test_tfidf.iloc[:, 1:]
test_y_tfidf = test_tfidf.iloc[:, 0]

eval_df_tfidf = pd.read_csv('project_data/data/tfidf/tfidf.eval.anon.csv')
eval_y_tfidf = eval_df_tfidf.iloc[:, 0]
eval_df_tfidf = eval_df_tfidf.iloc[:, 1:]



print("Loading Data")


train_y_tfidf = train_y_tfidf.replace(0, -1)
test_y_tfidf = test_y_tfidf.replace(0, -1)



def make_prediction(x, y, w, b):
    pred = np.dot(w.transpose(), x) + b
    if (pred > 0 and y == 1) or (pred < 0 and y == -1):
        return True
    else:
        return False


def calculate_accuracy(feature_df, label_df, w, b, flag=False):
    size = feature_df.shape[0]
    count = 0
    pred_list = []
    for i in range(size):
        x = feature_df.iloc[i].tolist()
        y = label_df.iloc[i]
        predicted_y = make_prediction(x, y, w, b)
        if flag:
            pred_list.append(predicted_y)
        if predicted_y:
            count += 1
    if flag:
        return (count / size), pred_list
    else:
        return (count / size)


def predictions(x, w, b):
    pred = np.dot(w.transpose(), x) + b
    if pred > 0:
        return 1
    else:
        return 0


def create_predicted_list(feature_df, w, b):
    pred_list = []
    size = feature_df.shape[0]
    count = 0
    for i in range(size):
        x = feature_df.iloc[i].tolist()
        pred_list.append(predictions(x, w, b))
    return pred_list


def initialize_parameter(size):
    np.random.seed(75)
    w = np.array(np.random.normal(-0.01, 0.01, size))
    b = np.random.normal(-0.01, 0.01)
    return w, b


def update_parameter(x, y, w, b, lr):
    eq = y * (np.dot(w.transpose(), x) + b)
    x = np.array(x)
    if eq < 0:
        w = w + lr * y * x
        b = b + lr * y
    return w, b


def compute_error(pred_list, D):
    error = 0
    for i in range(len(pred_list)):
        if pred_list[i] == False:
            error += D[i]
    return error


def calculate_alpha(error):
    return 0.5 * np.log((1 - error) / error)


def update_D(D, pred_list, alpha):
    for i in range(len(D)):
        if pred_list[i]:
            D[i] = D[i] * exp(-1 * alpha)
        else:
            D[i] = D[i] * exp(alpha)
    Z = np.sum(D)
    D = [D[i] / Z for i in range(len(D))]
    return D


def perceptron(features, labels, w, b, lr, epochs, dev=False, test_df_x=None, test_df_y=None):
    w_list = []
    b_list = []
    accuracies_list = []
    index_list = np.arange(features.shape[0])
    for e in range(epochs):
        print("epoch:", e)
        np.random.seed(e)
        np.random.shuffle(index_list)
        for i in index_list:
            x = features.iloc[i].tolist()
            y = labels.iloc[i]
            w, b = update_parameter(x, y, w, b, lr)
        if dev:
            b_list.append(b)
            w_list.append(w.copy())
            acc = calculate_accuracy(test_df_x, test_df_y, w, b)
            accuracies_list.append(acc)
            print("Developmental dataset accuracy for epoch", e, "=", acc)
    if dev:
        return w, b, accuracies_list, w_list, b_list
    else:
        return w, b


# In[39]:


def boosting_algorithm(train_x, train_y, lr, epochs):
    alphas = []
    weights = []
    bias = []
    N = train_x.shape[0]
    D = np.ones(N) / N

    for i in range(10):
        print("Round: ", i + 1)
        w, b = initialize_parameter(train_x.shape[1])
        w, b = perceptron(train_x, train_y, w, b, lr, epochs)
        acc, pred_list = calculate_accuracy(train_x, train_y, w, b, True)

        error = compute_error(pred_list, D)
        alpha = calculate_alpha(error)
        D = update_D(D, pred_list, alpha)

        bias.append(b)
        weights.append(w)
        alphas.append(alpha)

    return alphas, weights, bias


def predict_boosting(features, alphas, weights, bias):
    pred_list = []
    for i in range(features.shape[0]):
        total = 0
        x = features.iloc[i]
        for j in range(len(alphas)):
            a = alphas[j]
            w = weights[j]
            b = bias[j]

            total += a * (np.dot(w.transpose(), x) + b)
        if total < 0:
            pred = -1
        else:
            pred = 1
        pred_list.append(pred)
    return pred_list


def calculate_boosting_accuracy(pred_list, labels):
    count = 0
    total = len(pred_list)
    for i in range(total):
        if pred_list[i] == labels.iloc[i]:
            count += 1
    return count / total


train_length = int(train_x_tfidf.shape[0] * 4 // 5)
test_length = train_x_tfidf.shape[0] - train_length

train_x_fold = train_x_tfidf.head(train_length)
train_y_fold = train_y_tfidf.head(train_length)

test_x_fold = train_x_tfidf.tail(test_length)
test_y_fold = train_y_tfidf.tail(test_length)

lrs = [1, 0.1, 0.01]
accuracies = {}

for lr in lrs:
    print("lr:", lr)
    alphas, weights, bias = boosting_algorithm(train_x_fold, train_y_fold, lr, 10)
    pred_list = predict_boosting(test_x_fold, alphas, weights, bias)
    acc = calculate_boosting_accuracy(pred_list, test_y_fold)
    accuracies[lr] = acc
print(accuracies)



best_lr = max(accuracies, key=lambda x: accuracies[x])
lr = best_lr
print("best_lr:", lr)

alphas, weights, bias = boosting_algorithm(train_x_tfidf, train_y_tfidf, lr, 10)
pred_list = predict_boosting(train_x_tfidf, alphas, weights, bias)
acc = calculate_boosting_accuracy(pred_list, train_y_tfidf)
print("Train accuracy:", acc)
#print(len(test_x_tfidf))
pred_list = predict_boosting(test_x_tfidf, alphas, weights, bias)
acc = calculate_boosting_accuracy(pred_list, test_y_tfidf)
print("Dev accuracy:", acc)


predict = predict_boosting(eval_df_tfidf, alphas, weights, bias)


file = open('Boosted_tfidf.csv', 'w', newline='')

with file:
    header = ['example_id', 'label']
    writer = csv.DictWriter(file, fieldnames=header)

    writer.writeheader()
    for i in range(len(predict)):
        if predict[i] == -1:
            predict[i] = 0
        writer.writerow({'example_id': i, 'label': predict[i]})

# In[ ]:
eval_acc = calculate_boosting_accuracy(predict, eval_y_tfidf)
print(f'Test accuracy: {eval_acc}')



