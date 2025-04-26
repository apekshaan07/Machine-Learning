import util
from lr import decay_lr
from signum import sign
from perceptron import Perceptron
import csv

import numpy as np
import math
from   random import randint
import matplotlib.pyplot as plt


epoch_test = 20



np.random.seed(350)
PATH = 'project_data/data/tfidf/'
num_folds = 5

data_cv = []
labels= []
colr = 0

for i in range(num_folds):
	_, _, colr= util.file(PATH + 'tfidf.train.csv', colr)
	
for i in range(num_folds):
	data_fold, label_fold, colr= util.file(PATH + 'tfidf.train.csv', colr)
	data_cv.append (data_fold)
	labels.append(label_fold)

PATH  = 'project_data/data/tfidf/'
data_tr, label_tr, colr= util.file(PATH + 'tfidf.train.csv', colr)
data_evl, label_evl, colr= util.file(PATH + 'tfidf.eval.anon.csv'  , colr)
data_te, label_te, colr= util.file(PATH  + 'tfidf.test.csv' , colr)

label_tr[label_tr == 0] = -1
label_evl[label_evl == 0] = -1
label_te[label_te == 0] = -1

learning_rate = [1, 0.1, 0.01]
accuracy = np.zeros((len(learning_rate),num_folds))

for i in range(len(learning_rate)):

	
	for j in range(num_folds):

		if(j==0):
			start = 1			
			data_train = data_cv[1]
			label_train = labels[1]

			data_test  = data_cv[0]
			label_test = labels[1]
		else:
			start = 0
			data_train = data_cv[0]
			label_train = labels[0]

			data_test  = data_cv[j]
			label_test = labels[j]
	


		for k in range(start+1,num_folds):
			if(k != j):		
				data_train  = np.concatenate([data_train,  data_cv[k]] , axis=0)
				label_train = np.concatenate([label_train, labels[k]], axis=0)

		avgPerceptron = Perceptron(colr,use_flag = True)
		avgPerceptron.initialize_random()
		
		t = 0
		
		
		
		test_predict = avgPerceptron.predict(data_test)
		accuracy[i][j]    = util.accuracy(label_test, test_predict)	
		
mean = np.mean(accuracy,axis=1)
std_dev = np.std (accuracy,axis=1)

print("*************************** Average Perceptron*****************************")
print("learning_rate \t Accuracy_mean \t Accuracy_std")

for i in range(len(learning_rate)):
	print("%.2f \t %.2f \t\t %.2f" %(learning_rate[i],mean[i],std_dev[i]))


index = np.argwhere(mean==np.max(mean))
learning_rate_best = learning_rate[index[0,0]]
# learning_rate_best = learning_rate[np.argmax(mean)]

print("Best learning rate = " + str(learning_rate_best))


avgPerceptron = Perceptron(colr,use_flag = True)
avgPerceptron.initialize_random()
num_update = 0
accuracy_dev_best = 0
accuracy_dev_list = []
accuracy_test_list = []

t=0

for k in range(epoch_test):
	for l in range(label_tr.shape[0]):
		x = data_tr[l]
		y = label_tr[l]
		learning_rate_t = learning_rate_best
		if(avgPerceptron.predict_train(x)*y <= 0):
			avgPerceptron.update(learning_rate_best, x, y)
			num_update += 1
		avgPerceptron.update_avg()	

	
	predict_dev = avgPerceptron.predict(data_evl)
	accuracy_dev = util.accuracy(label_evl, predict_dev)
	accuracy_dev_list.append(accuracy_dev)
	predict_dev[predict_dev==0]=-1

	predict_te = avgPerceptron.predict(data_te)
	accuracy_test = util.accuracy(label_te, predict_te)
	accuracy_test_list.append(accuracy_test)

print("Total number of updates = " + str(num_update))

accuracy_dev_list = np.array(accuracy_dev_list)


index = np.argmax(accuracy_dev_list)
print("\nBest hyperparameter Accuracy\ndev  = %.2f\ntest = %.2f" %(accuracy_dev_list[index], accuracy_test_list[index]))



label_tr[label_tr == -1] = 0
label_evl[label_evl == -1] = 0
label_te[label_te == -1] = 0
predict_dev[predict_dev==-1]=0







file = open('tfidf_preceptron.csv', 'w', newline ='')
with file:
    header = ['example_id', 'label']
    writer = csv.DictWriter(file, fieldnames = header)

    writer.writeheader()
    for i in range(len(predict_dev)):
        writer.writerow({'example_id' : i, 'label': predict_dev[i]})


#output_csv_path = 'predicted_labels.csv'

# Write the predictions to a CSV file
#with open(output_csv_path, 'w', newline='') as csvfile:
 #   writer = csv.writer(csvfile)
  #  writer.writerow(['PredictedLabel'])  # Writing a header for the CSV file
   # for label in predict_te:
    #    writer.writerow([label])

