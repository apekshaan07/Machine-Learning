import numpy as np
from cal import sign


class SVM(object):
	def __init__(self, dim = 0, C = 0):
		self.dim      = dim + 1            # #dimensions of perceptron + bias
		self.w        = np.zeros(self.dim) # weights
		self.C        = C                  # Regularisation Hyperparameter

	def get_weight(self):
		return self.w

	def init_random(self):
		self.w = 0.02*np.random.rand(self.dim) - 0.01 # small number between -0.01 and 0.01
	
	def init_zeros(self):
		self.w = np.zeros(self.w.shape)
	
	def update(self, lr, x, y):
		pred = self.dot_with_weight(x)
		
		# Update assumes single example is passed		
		if (pred[0]*y <= 1):	
			self.w = (1 - lr) * self.w + lr * self.C * y * np.append(x,1)
		else:
			self.w = (1 - lr) * self.w
	
	def predict(self, x):
		if(x.ndim == 1):	
			return sign(np.array([self.w.dot(np.append(x,1))]))
		else:
			return sign(np.append(x,np.ones([len(x), 1]),1).dot(self.w.T))

	# This is required in updation step
	def dot_with_weight(self, x):
		if(x.ndim == 1):	
			return np.array([self.w.dot(np.append(x,1))])
		else:
			return np.append(x,np.ones([len(x), 1]),1).dot(self.w.T)
		
	def compute_loss(self, data, labels):
		loss = 0.0
		for i in range(len(data)):
			prediction = np.dot(self.weights, data[i])
			loss += max(0, 1 - labels[i] * prediction)
		return loss / len(data) + self.C * np.sum(self.weights**2)

'''
class SVM:
    def __init__(self, feature_count, C):
        self.C = C
        self.weights = np.zeros(feature_count)
    def init_zeros(self):
        # Optional: Explicitly re-initialize to zeros if needed outside the constructor
        self.weights = np.zeros_like(self.weights)

    def update(self, learning_rate, feature, label):
        decision = np.dot(self.weights, feature)
        if label * decision < 1:
            self.weights += learning_rate * (label * feature - 2 * self.C * self.weights)
        else:
            self.weights -= learning_rate * 2 * self.C * self.weights

        # Clipping weights to avoid overflow/underflow
        self.weights = np.clip(self.weights, -1e5, 1e5)
    def predict(self, features):
        """Make predictions with the current weights on the features."""
        return np.sign(np.dot(features, self.weights))

    def compute_loss(self, data, labels):
        loss = 0.0
        for i in range(len(data)):
            prediction = np.dot(self.weights, data[i])
            loss += max(0, 1 - labels[i] * prediction)
        return loss / len(data) + self.C * np.sum(self.weights**2)


    '''