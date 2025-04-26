import numpy as np
import math

'''
def sign(input):
	output = input.copy()
	
	output[output>=0] = 1
	output[output< 0] = -1
	
	return output
'''

def sign(input_array):
    return np.where(input_array >= 0, 1, -1)

def sigmoid(input):
	output = input.copy()

	z = np.exp(-output[output >= 0])
	output[output >= 0] = 1 / (1 + z)

	z = np.exp(output[output <   0])	
	output[output <  0] = z / (1 + z)

	return output