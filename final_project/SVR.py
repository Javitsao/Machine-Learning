import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

train_data = pd.read_csv("train_data.csv", usecols=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Likes', 'Composer', 'Artist'])
# train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Tempo', 'Composer', 'Artist'])
print(type(train_data))

column_averages = train_data.iloc[:, :16].mean(skipna=True)
train_data = train_data[(train_data['Danceability'] >= 0) & (train_data['Danceability'] <= 9)]
# print(train_data)
column_medians = train_data.iloc[:, :16].median()
train_data.iloc[:, :16] = train_data.iloc[:, :16].fillna(column_medians)


X = train_data.iloc[:, 1:16]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column
X = X.values
y = y.values.ravel()
print(type(X))
print(type(y))
# print(X, y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=24)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

svr_regressor = SVR(kernel='rbf', C=0.17, epsilon=0.5)

# Predict danceability on the validation set
svr_regressor.fit(X_train, y_train)

# Predict danceability on the validation set
y_pred = svr_regressor.predict(X_val)

# Calculate accuracy of the model
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)

real_y_Ein = svr_regressor.predict(X_train)
for i in range(len(real_y_Ein)):
    if real_y_Ein[i] < 0:
        real_y_Ein[i] = 0
    if real_y_Ein[i] > 9:
        real_y_Ein[i] = 9
real_Ein = mean_absolute_error(y_train, real_y_Ein)
print("Real Ein:", real_Ein)

test_data = pd.read_csv("test_data.csv", usecols=['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Likes', 'Composer', 'Artist'])
# test_data = pd.read_csv("test.csv", usecols=['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Tempo', 'Composer', 'Artist'])
column_averages = test_data.iloc[:, :15].mean(skipna=True)
column_medians2 = test_data.iloc[:, :15].median()
test_data.iloc[:, :15] = test_data.iloc[:, :15].fillna(column_averages)
# print(test_data)

X = scaler.transform(X)
y_Ein = svr_regressor.predict(X)
for i in range(len(y_Ein)):
    if y_Ein[i] < 0:
        y_Ein[i] = 0
    if y_Ein[i] > 9:
        y_Ein[i] = 9
# for i in range(len(y_Ein)):
#     y_Ein[i] = round(y_Ein[i])
Ein = mean_absolute_error(y, y_Ein)
print("Ein:", Ein)

test_data = test_data.values
# print(test_data)
test_data = scaler.transform(test_data)
# y_pred = logreg.predict(test_data)
ans = svr_regressor.predict(test_data)
np.set_printoptions(threshold=np.inf)
# print(ans)

for i in range(len(ans)):
    if ans[i] < 0:
        ans[i] = 0
    if ans[i] > 9:
        ans[i] = 9
    ans[i] = round(ans[i])

# os.remove("./SVR.csv")
csv_file = "SVR.csv"

# Generate the index values starting from 17170
index_values = range(17170, 17170 + len(ans))

# Write the predicted danceability values to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'Danceability'])
    writer.writerows(zip(index_values, ans))

print("CSV file 'SVR.csv' created successfully.")
