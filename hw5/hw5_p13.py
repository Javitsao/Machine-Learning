import libsvm.svmutil as lib
import numpy as np
train_y, train_x = lib.svm_read_problem('train.txt')
test_y, test_x = lib.svm_read_problem('test.txt')

# xx = sum([a * np.array([b[i] for i in sorted(b.keys())]) for a, b in zip(x.get_sv_coef(), x.get_SV())])
# normal = np.sqrt((np.array(xx) ** 2).sum())
maxi = -1000000
rec = []
for n in [2, 3, 4, 5, 6]:
    x = (lib.svm_train([int(i != n) for i in train_y], train_x, f'-s 0 -h 0 -q -c 1 -t 1 -d 2 -r 1 -g 1' ))
    rec.append(len(x.get_SV()))
    
print(min(rec))