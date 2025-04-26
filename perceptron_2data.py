

import numpy as np
import csv
from perceptron import Perceptron  
import util


np.random.seed(42)


train_path_1 = 'project_data/data/tfidf/tfidf.train.csv'
train_path_2 = 'project_data/data/bag-of-words/bow.train.csv'
eval_path_1 = 'project_data/data/tfidf/tfidf.eval.anon.csv'
eval_path_2 = 'project_data/data/bag-of-words/bow.eval.anon.csv'
test_path_1 = 'project_data/data/tfidf/tfidf.test.csv'
test_path_2 = 'project_data/data/bag-of-words/bow.test.csv'


train_data, train_labels, _ = util.file(train_path_1, train_path_2)
eval_data, eval_labels, _ = util.file(eval_path_1, eval_path_2)
test_data_1, test_labels_1, _ = util.file(test_path_1)
test_data_2, test_labels_2, _ = util.file(test_path_2)


train_labels[train_labels == 0] = -1
eval_labels[eval_labels == 0] = -1
test_labels_1[test_labels_1 == 0] = -1
test_labels_2[test_labels_2 == 0] = -1


combined_test_data = np.concatenate((test_data_1, test_data_2), axis=0)
combined_test_labels = np.concatenate((test_labels_1, test_labels_2), axis=0)


perceptron = Perceptron(dime=train_data.shape[1], use_flag=True)
perceptron.initialize_random()


learning_rate = 0.1
epochs = 10


for epoch in range(epochs):
    for i, (x, y) in enumerate(zip(train_data, train_labels)):
        pred = perceptron.predict_train(x)
        if pred != y:
            perceptron.update(learning_rate, x, y)
            perceptron.update_avg()


train_predictions = perceptron.predict(train_data)
train_accuracy = util.accuracy(train_labels, train_predictions)
print(f"Training Accuracy: {train_accuracy}%")


test_predictions = perceptron.predict(combined_test_data)
test_accuracy = util.accuracy(combined_test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy}%")


eval_predictions = perceptron.predict(eval_data)
eval_accuracy = util.accuracy(eval_labels, eval_predictions)
print(f"Evaluation Accuracy: {eval_accuracy}%")


eval_predictions[eval_predictions == -1] = 0


output_path = 'perceptron_tfidf_bow.csv'
with open(output_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['example_id', 'label'])
    for i, label in enumerate(eval_predictions):
        writer.writerow([i, int(label)])

print(f"Eval predictions have been written to {output_path}")
