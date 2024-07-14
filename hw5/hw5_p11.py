import libsvm.svmutil as lib
import numpy as np
train_y, train_x = lib.svm_read_problem('train.txt')
test_y, test_x = lib.svm_read_problem('test.txt')

x = (lib.svm_train([int(i != 1) for i in train_y], train_x, f'-s 0 -h 0 -q -c 1 -t 0' ))
xx = sum([a * np.array([b[i] for i in sorted(b.keys())]) for a, b in zip(x.get_sv_coef(), x.get_SV())])
print(np.sqrt((np.array(xx) ** 2).sum()))