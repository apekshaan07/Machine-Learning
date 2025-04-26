import numpy as np
def sign(input):
    output = np.where(input >= 0, 1, -1)
    return output