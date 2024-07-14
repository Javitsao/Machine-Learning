import numpy as np
from sklearn.datasets import load_svmlight_file

# Load the training data
X_train, y_train = load_svmlight_file("train.txt")
X_train = X_train.toarray()

# Load the testing data
X_test, y_test = load_svmlight_file("test.txt")
X_test = X_test.toarray()

# Convert the labels to binary: 1 for label "11" and -1 for label "26"
positive_class = 11
negative_class = 26
X_train = X_train[(y_train == positive_class) | (y_train == negative_class)]
y_train_binary = y_train[(y_train == positive_class) | (y_train == negative_class)]
y_train_binary = np.where(y_train_binary == positive_class, 1, -1)

# Create a binary testing dataset for the one-versus-one problem
X_test = X_test[(y_test == positive_class) | (y_test == negative_class)]
y_test_binary = y_test[(y_test == positive_class) | (y_test == negative_class)]
y_test_binary = np.where(y_test_binary == positive_class, 1, -1)

# Initialize weights
N = len(X_train)
w = np.full(N, 1 / N)

# Store the selected weak classifiers
classifiers = []

# AdaBoost-Stump algorithm
for t in range(1000):
    # Normalize weights
    #w /= np.sum(w)
    
    # Find the best decision stump
    best_err = float('inf')
    best_s = 0
    best_feature = 0
    best_theta = 0
    for i in range(X_train.shape[1]):
        idx = np.argsort(X_train[:, i])
        X_sort = X_train[idx]
        sorted_labels = y_train_binary[idx]
        w_sort = w[idx]
        
        # Compute midpoints
        midpoints = (X_sort[:-1, i] + X_sort[1:, i]) / 2
        
        for s in [-1, 1]:
            for theta in midpoints:
                predict = s * np.where(X_sort[:, i] <= theta, 1, -1)
                err = np.sum(w_sort[predict != sorted_labels])
                #print(err)
                if err < best_err:
                    best_feature = i
                    best_theta = theta
                    best_err = err
                    best_s = s
                    best_idx = idx
    
    # Calculate a
    a = 0.5 * np.log((1 - best_err) / best_err)
    
    # Update weights
    X_sort = X_train[best_idx]
    sorted_labels = y_train_binary[best_idx]
    predict = best_s * np.where(X_sort[:, best_feature] <= best_theta, 1, -1)
    w *= np.exp(-a * predict * sorted_labels)
    #print(w)
    # Store the weak classifier
    classifiers.append((best_s, best_feature, best_idx, best_theta, a))

# Compute Ein(gt) for each weak classifier
rec_Ein = []
for classifier in classifiers:
    s, feature, idx, theta, a = classifier
    X_sort = X_train[idx]
    sorted_labels = y_train_binary[idx]
    predict = s * np.where(X_sort[:, feature] <= theta, 1, -1)
    Ein = np.mean(np.sign(predict) != sorted_labels)
    rec_Ein.append(Ein)

# Calculate min1≤1000≤t Ein(gt)
min_Ein = min(rec_Ein)

print("Minimum Ein(gt):", min_Ein)
