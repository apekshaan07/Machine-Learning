
import numpy as np


def decay_lr(base_lr, t):
	return float(base_lr)/(1+t)

#def aggressive_lr(mu, x, y, w):
#	return float(mu - y * w.dot(np.append(x,1)))/(1+x.dot(x)+1)

def aggressive_lr(mu, x, y, w):
   
    x_augmented = np.append(x, 1)  
   
    dot_product = np.dot(w, x_augmented)
    
    numerator = mu - y * dot_product
    
    denominator = np.dot(x_augmented, x_augmented) + 1

    return numerator / denominator