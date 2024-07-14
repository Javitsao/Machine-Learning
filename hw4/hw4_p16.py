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

data_split_train = np.zeros((120, 11), dtype=float)
data_split_test = np.zeros((80, 11), dtype=float)
x_split_train = np.zeros((120, 1001), dtype=float)
x_split_test = np.zeros((80, 1001), dtype=float)
x_split_train[:, 0] = 1
x_split_test[:, 0] = 1
times = [0, 0, 0, 0, 0]
E_sum = 0
for iter in range(256):
    random.seed(iter)
    #selected_rows = random.sample(list(data_train), 120)
    train_indices = np.random.choice(200, size=120, replace=False)
    data_split_train = data_train[train_indices, :]
    x_split_train[:, 1:11] = data_split_train[:, 0:10]
    y_split_train = data_train[train_indices, 10]
    #random.seed(iter)
    #selected_rows = random.sample(list(data_train), 80)
    test_indices = np.setdiff1d(np.arange(200), train_indices)
    data_split_test = data_train[test_indices, :]
    x_split_test[:, 1:11] = data_split_test[:, 0:10]
    y_split_test = data_train[test_indices, 10]
    rec = 11
    for i in range(0, 10):
        for j in range(i, 10):
            x_split_test[:, rec] = data_split_test[:, i] * data_split_test[:, j]
            x_split_train[:, rec] = data_split_train[:, i] * data_split_train[:, j]
            rec += 1
    for i in range(0, 10):
        for j in range(i, 10):
            for k in range(j, 10):
                x_split_test[:, rec] = data_split_test[:, i] * data_split_test[:, j] * data_split_test[:, k]
                x_split_train[:, rec] = data_split_train[:, i] * data_split_train[:, j] * data_split_train[:, k]
                rec += 1
    for i in range(0, 10):
        for j in range(i, 10):
            for k in range(j, 10):
                for l in range(k, 10):
                    x_split_test[:, rec] = data_split_test[:, i] * data_split_test[:, j] * data_split_test[:, k] * data_split_test[:, l]
                    x_split_train[:, rec] = data_split_train[:, i] * data_split_train[:, j] * data_split_train[:, k] * data_split_train[:, l]
                    rec += 1
    arr = [-6, -3, 0, 3, 6]
    max = 0.0
    for j in range(5):
        C = 1 / (2 * 10**arr[j])
        #print(C)
        s = '-q -s 0 -e 0.000001 -c '
        s += str(C)
        prob = problem(y_train, x_train)
        param = parameter(s)
        m = train(prob, param)
        #one, two, three = predict(y_test, x_test, m)
        one, two, three = predict(y_split_train, x_split_train, m)
        # print(one)
        # print(two)
        # print(three)
        if two[0] >= max:
            max = two[0]
            index = j
    C = 1 / (2 * 10**arr[index])
    #print(C)
    s = '-q -s 0 -e 0.000001 -c '
    s += str(C)
    prob = problem(y_train, x_train)
    param = parameter(s)
    m = train(prob, param)
    #one, two, three = predict(y_test, x_test, m)
    one, two, three = predict(y_test, x_test, m)
    E_sum += (100 - two[0]) / 100
    #print(arr[rec_j])
    #print(max)
print(E_sum / 256)
