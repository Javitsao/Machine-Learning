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
#print(rec)
arr = [-6, -3, 0, 3, 6]
max = 0.0
for i in range(5):
    C = 1 / (2 * 10**arr[i])
    #print(C)
    s = '-q -s 0 -e 0.000001 -c '
    s += str(C)
    prob = problem(y_train, x_train)
    param = parameter(s)
    m = train(prob, param)
    one, two, three = predict(y_test, x_test, m)
    #one, two, three = predict(y_train, x_train, m)
    # print(one)
    # print(two)
    # print(three)
    if two[0] >= max:
        max = two[0]
        rec_i = i
C = 1 / (2 * 10**arr[rec_i])
#print(C)
s = '-q -s 0 -e 0.000001 -c '
s += str(C)
prob = problem(y_train, x_train)
param = parameter(s)
m = train(prob, param)
count = 0
w = m.get_decfun()[0]
for i in range(len(w)):
    if w[i] <= 0.000001 and w[i] >= -0.000001:
        count += 1
print(count)
#print(max)
