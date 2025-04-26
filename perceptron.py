import numpy as np
from signum import sign




import numpy as np

def sign(x):
    return np.sign(x)

class Perceptron(object):
    def __init__(self, dime=0, use_flag=False):
        self.dime = dime + 1
        self.w = np.zeros(self.dime)
        self.w_average = np.zeros(self.dime)
        self.use_flag = use_flag
        self.count = 0

    def get_weight(self):
        return self.w

    def initialize_random(self):
        self.w = 0.02 * np.random.rand(self.dime) - 0.01
        self.w_average = self.w.copy()

    def update(self, learning_rate, x, y):
        x = np.array(x)
        self.w += learning_rate * y * np.append(x, 1)

    def update_avg(self):
        self.count += 1
        if self.use_flag:
            alpha = 1.0 / self.count

            self.w_average = (1 - alpha) * self.w_average + alpha * self.w


    def predict(self, x):
        x = np.array(x)
        if x.ndim == 1:
            x = np.append(x, 1)
        else:
            x = np.append(x, np.ones((x.shape[0], 1)), axis=1)
        
        if self.use_flag:
            raw_scores = x.dot(self.w_average.T)
        else:
            raw_scores = x.dot(self.w.T)
        return sign(raw_scores)
    
    def predict_train(self, x):
        if x.ndim == 1:
            return sign(np.array([self.w.dot(np.append(x,1))]))
        else:
            return sign(np.append(x,np.ones([len(x),1]),1).dot(self.use_flag.T))





    


