import numpy as np
import random
import math
from sklearn.linear_model import LinearRegression
# Load the data from the text file
data = np.loadtxt('train.txt', dtype= float)
x = np.zeros((100, 11), dtype=float)
x[:, 1:11] = data[:, 0:10]
y = data[:, 10]
x[:, 0] = 1

data = np.loadtxt('out.txt', dtype=float)
x_o = np.zeros((400, 11), dtype=float)
x_o[:, 1:11] = data[:, 0:10]
y_o = data[:, 10]
x_o[:, 0] = 1

eta = 0.001
rec_ein = 0
rec_eout = 0
rec_e = 0
#rec_ein[:, 0] = 100000
#w = np.linalg.pinv

for i in range(1000):
    w = np.array([0.29070963, -0.04988084, 0.04893561, -0.08623605, -0.06658103, 0.10689752, -0.12356574, 0.09486241, 0.26696655, -0.15660245, -0.06382855])
    random.seed(i)
    ein = 0
    eout = 0
    for j in range(800):
        index = random.randint(0, 99)
        #w -= eta * 2 * (np.matmul(np.transpose(x[index]), x[index]) * w - np.transpose(x[index]) * y[index])
        w += eta * (1/(1 + math.exp(y[index] * np.matmul(np.transpose(w), x[index])))) * y[index] * x[index]
        
    for j in range(100):
        #ein += (np.matmul(np.transpose(w), x[j]) - y[j]) ** 2
        if(np.matmul(np.transpose(w), x[j]) * y[j] < 0):
            ein += 1
    for j in range(400):
        if(np.matmul(np.transpose(w), x_o[j]) * y_o[j] < 0):
            eout += 1
    
    rec_e += abs(ein / 100 - eout / 400)
print(rec_e / 1000)