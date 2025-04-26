import numpy as np
import math
from random import randint


def file(path, max_col_prior=0):
    matrix = []
    label = []

    max_col_cnt = 0

    with open(path) as f:
        next(f)  
        for line in f:
            
            data = line.strip().split(',')  
            try:
                label_value = float(data[0])  
                label.append(label_value)
            except ValueError:
                
                print(f"Skipping invalid data line: {line}")
                continue
            
            row = []
            col_cnt = 0
            for item in data[1:]:  
                if ':' in item:  
                    idx, value = item.split(':')
                   
                    n = int(idx) - (col_cnt + 1)
                    row.extend([0] * n)  
                else:
                    value = item  
                row.append(float(value))
                col_cnt += 1
            
            matrix.append(row)
            max_col_cnt = max(max_col_cnt, col_cnt)

    
    for row in matrix:
        if len(row) < max_col_cnt:
            row.extend([0] * (max_col_cnt - len(row)))

    return np.array(matrix), np.array(label), max_col_cnt



def accuracy(labels, predicted):
    if labels.shape != predicted.shape:
        print("Array sizes do not match")
        return 0.0  
    correct = np.sum(labels == predicted)

    accuracy = (correct / labels.size) * 100
    return accuracy
