import libsvm.svmutil as lib
import numpy as np
train_y, train_x = lib.svm_read_problem('train.txt')
test_y, test_x = lib.svm_read_problem('test.txt')
def Cal(m, x, y):
    p_label, p_acc, p_val = lib.svm_predict(y, x, m, '-q')
    ACC, MSE, SCC = lib.evaluations(y, p_label)
    return 1 - ACC / 100

# xx = sum([a * np.array([b[i] for i in sorted(b.keys())]) for a, b in zip(x.get_sv_coef(), x.get_SV())])
# normal = np.sqrt((np.array(xx) ** 2).sum())
rec = []
for c in [0.01, 0.1, 1, 10, 100]:
    x = (lib.svm_train([int(i != 7) for i in train_y], train_x, f'-s 0 -h 0 -q -c {c} -t 2 -g 1' ))
    rec.append(([Cal(x, test_x, [int(i != 7) for i in test_y])], c))
    
print(min(rec)[1])