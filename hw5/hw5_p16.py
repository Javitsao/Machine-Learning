import libsvm.svmutil as lib
import numpy as np
train_y, train_x = lib.svm_read_problem('train.txt')
test_y, test_x = lib.svm_read_problem('test.txt')
def E(m, x, y):
    p_label, p_acc, p_val = lib.svm_predict(y, x, m, '-q')
    ACC, MSE, SCC = lib.evaluations(y, p_label)
    return 1 - ACC/100
train_x = np.array(train_x)
train_y = np.array(train_y)
v = dict()
for i in range(500):
    ans = (2, 0)
    for g in [0.1, 1, 10, 100, 1000]:
        idx = np.arange(len(train_x))
        np.random.shuffle(idx)
        trn_i, val_i = idx[200:], idx[:200]
        m = lib.svm_train([int(i != 7) for i in train_y[trn_i]], train_x[trn_i], f'-s 0 -h 0 -q -c 0.1 -t 2 -g {g}')
        e = E(m , train_x[val_i], train_y[val_i])
        ans = min(ans, (e, g))
        print(ans)
    v[ans[1]] = v.get(ans[1], 0) + 1
    
print(max([(v[i], i) for i in v])[1])
