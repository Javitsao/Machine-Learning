import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
# Load the data from the text file
data = np.loadtxt('train.txt', dtype= float)
x = np.zeros((100, 21), dtype=float)
x[:, 1:11] = data[:, 0:10]
x[:, 11:21] = np.square(data[:, 0:10])
y = data[:, 10]
x[:, 0] = 1

data = np.loadtxt('out.txt', dtype=float)
x_o = np.zeros((400, 21), dtype=float)
x_o[:, 1:11] = data[:, 0:10]
x_o[:, 11:21] = np.square(data[:, 0:10])
y_o = data[:, 10]
x_o[:, 0] = 1

eta = 0.001
rec_ein = 0
rec_eout = 0
rec_e = 0
#rec_ein[:, 0] = 100000
w = np.matmul(np.matmul(np.linalg.pinv(np.matmul(np.transpose(x), x)), np.transpose(x)), y)

for i in range(1000):
    random.seed(i)
    ein = 0
    eout = 0
    
    for j in range(100):
        #ein += (np.matmul(np.transpose(w), x[j]) - y[j]) ** 2
        if(np.matmul(np.transpose(w), x[j]) * y[j] < 0):
            ein += 1
    for j in range(400):
        if(np.matmul(np.transpose(w), x_o[j]) * y_o[j] < 0):
            eout += 1
    
    rec_e += abs(ein / 100 - eout / 400)
print(rec_e / 1000)