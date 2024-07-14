import numpy as np
from sklearn.linear_model import LinearRegression
# Load the data from the text file
data = np.loadtxt('train.txt', dtype= float)
x = np.zeros((100, 11))
x[:, 1:11] = data[:, 0:10]
y = data[:, 10]
x[:, 0] = 1
print(x)
# Fit the linear regression model
reg = LinearRegression().fit(x, y)

# Print the coefficient of determination (R^2)
r2 = reg.score(x, y)
print("Coefficient of determination (R^2):", r2)

# Compute the in-sample error (Ein)
y_pred = reg.predict(x)
ein = np.mean((y - y_pred) ** 2)
print("In-sample error (Ein):", ein)
