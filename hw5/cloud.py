import libsvm.svmutil as lib
import numpy as np

OPT_LINEAR = 0
OPT_POLYNOMIAL = 1
OPT_RADIAL_BASIS_FUNCTION = 2
OPT_SIGMOID = 3
OPT_PRECOMPUTED_KERNEL = 4

def params(**d): return f'-s 0 -h 0 -q' + ''.join([f' -{k} {v}' for k, v in d.items()])
def sign(x): return np.where(x >= 0, 1, -1)
def E(m, x, y):
    p_label, p_acc, p_val = lib.svm_predict(y, x, m, '-q')
    ACC, MSE, SCC = lib.evaluations(y, p_label)
    return 1 - ACC/100
def to_arr(x): return np.array([x[i] for i in sorted(x.keys())])
def W(m): return sum([a * to_arr(b) for a, b in zip(m.get_sv_coef(), m.get_SV())])
def norm(x): return np.sqrt((np.array(x) ** 2).sum())
def rand_idx(n):
    x = np.arange(n)
    np.random.shuffle(x)
    return x

def target(y, c):
    return [int(i != c) for i in y]

def P11(train_x, train_y, test_x, test_y):
    print(norm(W(lib.svm_train(target(train_y, 1), train_x, params(c=1, t=OPT_LINEAR)))))

def P12(train_x, train_y, test_x, test_y):
    print(max([(E(
        lib.svm_train(target(train_y, n), train_x, params(c=1, t=OPT_POLYNOMIAL, d=2, r=1, g=1)),
        train_x,
        target(train_y, n),
    ), n) for n in [2, 3, 4, 5, 6]])[1])

def P13(train_x, train_y, test_x, test_y):
    print(min([
        len(lib.svm_train(target(train_y, n), train_x, params(c=1, t=OPT_POLYNOMIAL, d=2, r=1, g=1)).get_SV())
    for n in [2, 3, 4, 5, 6]]))

def P14(train_x, train_y, test_x, test_y):
    print(min([(E(
        lib.svm_train(target(train_y, 7), train_x, params(c=c, t=OPT_RADIAL_BASIS_FUNCTION, g=1)),
        test_x,
        target(test_y, 7),
    ), c) for c in [0.01, 0.1, 1, 10, 100]])[1])

def P15(train_x, train_y, test_x, test_y):
    print(min([(E(
        lib.svm_train(target(train_y, 7), train_x, params(c=0.1, t=OPT_RADIAL_BASIS_FUNCTION, g=g)),
        test_x,
        target(test_y, 7),
    ), g) for g in [0.1, 1, 10, 100, 1000]])[1])

def P16(train_x, train_y, test_x, test_y):
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    vote = dict()
    for _ in range(500):
        ans = (2, 0)
        for g in [0.1, 1, 10, 100, 1000]:
            idx = rand_idx(len(train_x))
            trn_i, val_i = idx[200:], idx[:200]
            m = lib.svm_train(target(train_y[trn_i], 7), train_x[trn_i], params(c=0.1, t=OPT_RADIAL_BASIS_FUNCTION, g=g))
            e = E(m , train_x[val_i], train_y[val_i])
            ans = min(ans, (e, g))
        vote[ans[1]] = vote.get(ans[1], 0) + 1
    print(max([(vote[i], i) for i in vote])[1])


class DecisionStump:
    def __init__(self, train_x, train_y, u):
        self.N, = train_x.shape
        idx = np.argsort(train_x)
        self.train_x = train_x[idx]
        self.train_y = train_y[idx]
        self.u = u[idx]

        def theta(i):
            return -np.inf if i == 0 else (self.train_x[i-1] + self.train_x[i]) / 2
        cl = np.cumsum(self.u)
        cr = np.flip(np.cumsum(np.flip(self.u, 0)), 0)
        cpl = np.cumsum((self.train_y == 1) * self.u)
        cpr = np.flip(np.cumsum(np.flip((self.train_y == 1) * self.u, 0)), 0)
        def loss(s, i):
            # assert i == 0 or self.train_x[i] != self.train_x[i-1]
            lp = 0 if i == 0 else cpl[i-1]
            # assert np.isclose(lp, ((self.train_y[:i] == 1) * self.u[:i]).sum())
            ln = (0 if i == 0 else cl[i-1]) - lp
            # assert np.isclose(ln, ((self.train_y[:i] == -1) * self.u[:i]).sum())
            rp = cpr[i]
            # assert np.isclose(rp, ((self.train_y[i:] == 1) * self.u[i:]).sum())
            rn = cr[i] - rp
            # assert np.isclose(rn, ((self.train_y[i:] == -1) * self.u[i:]).sum())
            if s == -1: return (ln + rp)
            else: return (lp + rn)

        best = (np.inf, np.inf, np.inf)
        for i in range(len(self.train_x)):
            if i != 0 and self.train_x[i-1] == self.train_x[i]:
                continue
            best = min(best, (loss(+1, i), +1, theta(i)))
            best = min(best, (loss(-1, i), -1, theta(i)))

        self.e, self.s, self.t = best

    def Err(self):
        return (self.train_y != self.predict(self.train_x)).mean()

    def predict(self, x):
        return self.s * sign(x - self.t)

class MultiDimensionalDecisionStump:
    def __init__(self, train_x, train_y, u):
        ds = [DecisionStump(train_x[:, i], train_y, u) for i in range(train_x.shape[1])]
        _, self.i = min([(ds[i].e, i) for i in range(len(ds))])
        self.h = ds[self.i]
        self.s, self.t = self.h.s, self.h.t

    def Err(self):
        return self.h.Err()

    def predict(self, x):
        return self.h.predict(x[:, self.i])

class AdaBoost:
    def __init__(self, train_x, train_y, T=1000):
        self.train_x = train_x
        self.train_y = train_y
        self.T = T
        self.N = len(train_x)
        self.u = [np.ones((self.N,)) / self.N]
        self.g = [MultiDimensionalDecisionStump(train_x, train_y, self.u[-1])]
        self.a = [np.log(np.sqrt(1 / ((self.u[-1] * (train_y != self.g[-1].predict(train_x))).sum() / self.u[-1].sum()) - 1))]
        print(self.a)
        for _ in range(1, T):
            self.u.append(self.u[-1] * (np.exp(self.a[-1]) ** np.where(
                train_y != self.g[-1].predict(train_x), 1, -1
            )))
            self.g.append(MultiDimensionalDecisionStump(train_x, train_y, self.u[-1]))
            self.a.append(np.log(np.sqrt(1 / ((self.u[-1] * (train_y != self.g[-1].predict(train_x))).sum() / self.u[-1].sum()) - 1)))

        self.a = np.array(self.a)

    def predict(self, x):
        return sign((self.a @ np.array([i.predict(x) for i in self.g])))

    def Err(self, x=None, y=None):
        if x is None: x = self.train_x
        if y is None: y = self.train_y
        return (y != self.predict(x)).mean()

def P17(train_x, train_y, test_x, test_y):
    print(min([i.Err() for i in AdaBoost(train_x, train_y).g]))

def P18(train_x, train_y, test_x, test_y):
    print(max([i.Err() for i in AdaBoost(train_x, train_y).g]))

def P19(train_x, train_y, test_x, test_y):
    print(AdaBoost(train_x, train_y).Err())

def P20(train_x, train_y, test_x, test_y):
    print(AdaBoost(train_x, train_y).Err(test_x, test_y))

if __name__ == '__main__':
    train_y, train_x = lib.svm_read_problem('train.dat')
    test_y, test_x = lib.svm_read_problem('test.dat')

    P11(train_x, train_y, test_x, test_y)
    P12(train_x, train_y, test_x, test_y)
    P13(train_x, train_y, test_x, test_y)
    P14(train_x, train_y, test_x, test_y)
    P15(train_x, train_y, test_x, test_y)
    P16(train_x, train_y, test_x, test_y)

    train_x = np.array([to_arr(i) for i in train_x])
    train_y = np.array(train_y)
    test_x = np.array([to_arr(i) for i in test_x])
    test_y = np.array(test_y)
    train_x, train_y = train_x[(train_y == 11) | (train_y == 26)], train_y[(train_y == 11) | (train_y == 26)]
    train_y = np.where(train_y == 11, -1, 1)
    test_x, test_y = test_x[(test_y == 11) | (test_y == 26)], test_y[(test_y == 11) | (test_y == 26)]
    test_y = np.where(test_y == 11, -1, 1)

    P17(train_x, train_y, test_x, test_y)
    P18(train_x, train_y, test_x, test_y)
    P19(train_x, train_y, test_x, test_y)
    P20(train_x, train_y, test_x, test_y)
