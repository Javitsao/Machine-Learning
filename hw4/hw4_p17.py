import numpy as np
import random
import math
from liblinear.liblinearutil import *
# Load the data from the text file
data_train = np.loadtxt('train.txt', dtype= float)
data_test = np.loadtxt('test.txt', dtype= float)
x_train = np.zeros((200, 1001), dtype=float)
x_test = np.zeros((500, 1001), dtype=float)
y_train = data_train[:, 10]
y_test = data_test[:, 10]
x_train[:, 0] = 1
x_train[:, 1:11] = data_train[:, 0:10]
x_test[:, 0] = 1
x_test[:, 1:11] = data_test[:, 0:10]
rec = 11
for i in range(0, 10):
    for j in range(i, 10):
        x_test[:, rec] = data_test[:, i] * data_test[:, j]
        x_train[:, rec] = data_train[:, i] * data_train[:, j]
        rec += 1
for i in range(0, 10):
    for j in range(i, 10):
        for k in range(j, 10):
            x_test[:, rec] = data_test[:, i] * data_test[:, j] * data_test[:, k]
            x_train[:, rec] = data_train[:, i] * data_train[:, j] * data_train[:, k]
            rec += 1
for i in range(0, 10):
    for j in range(i, 10):
        for k in range(j, 10):
            for l in range(k, 10):
                x_test[:, rec] = data_test[:, i] * data_test[:, j] * data_test[:, k] * data_test[:, l]
                x_train[:, rec] = data_train[:, i] * data_train[:, j] * data_train[:, k] * data_train[:, l]
                rec += 1

data_split_train = np.zeros((160, 1001), dtype=float)
data_split_test = np.zeros((40, 1001), dtype=float)
x_split_train = np.zeros((160, 1001), dtype=float)
x_split_test = np.zeros((40, 1001), dtype=float)
x_split_train[:, 0] = 1
x_split_test[:, 0] = 1
times = [0, 0, 0, 0, 0]
E_sum = 0
lamda = [-6, -3, 0, 3, 6]
for iter in range(256):
    random.seed(iter)
    #selected_rows = random.sample(list(data_train), 80)
    idx = random.sample(range(0, 200), 200)
    data_train = data_train[idx]
    min = 100
    for j in range(5):
        C = 1 / (2 * 10**lamda[j])
        #print(C)
        ECV = 0
        for fold in range(5):
            #f = random.randint(0, 4)
            arr = (40 * fold <= np.arange(200)) & (np.arange(200) < 40 * (fold + 1))
            x_split_test = x_train[arr, :]
            y_split_test = y_train[arr]
            x_split_train = x_train[~arr, :]
            y_split_train = y_train[~arr]
            
            max = 0.0
            s = '-q -s 0 -e 0.000001 -c '
            s += str(C)
            prob = problem(y_split_train, x_split_train)
            param = parameter(s)
            m = train(prob, param)
            #one, two, three = predict(y_test, x_test, m)
            one, two, three = predict(y_split_test, x_split_test, m)
            # print(one)
            # print(two)
            # print(three)
            ECV += (100 - two[0]) / 100
        if ECV < min:
            min = ECV
        #print(ECV)
    
    E_sum += min
    #print(E_sum / (iter + 1) / 5)
print(E_sum / 256 / 5)
