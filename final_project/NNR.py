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
from sklearn.model_selection import cross_val_score
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
E=3
train_data = pd.read_csv("train_data.csv", usecols=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Likes', 'Composer', 'Artist'])
# train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Tempo', 'Composer', 'Artist'])
print(type(train_data))

column_averages = train_data.iloc[:, :18].mean(skipna=True)
train_data = train_data[(train_data['Danceability'] >= 0) & (train_data['Danceability'] <= 9)]
# print(train_data)
column_medians = train_data.iloc[:, :18].median()
train_data.iloc[:, :18] = train_data.iloc[:, :18].fillna(column_medians)


X = train_data.iloc[:, 1:18]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column
mean_y = y.mean()  # Calculate the mean of 'y'
print("Mean of y:", mean_y)
X = X.values
y = y.values.ravel()
print(type(X))
print(type(y))
# print(X, y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=99)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# X_train = np.array(X_train)
# y_train = np.array(y_train)

model = Sequential()

# Add input layer and hidden layers
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.2))  # Dropout layer with 20% dropout rate
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Dropout layer with 20% dropout rate
model.add(Dense(1))  # Output layer

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=E, batch_size=32, validation_data=(X_val, y_val))

# # Perform cross-validation
# cv_scores = cross_val_score(svr_regressor, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

# # Calculate the mean absolute error across cross-validation folds
# mae_cv = -cv_scores.mean()

# print("Cross-Validated Mean Absolute Error:", mae_cv)

# Predict danceability on the validation set
model.fit(X_train, y_train)

# Predict danceability on the validation set
y_pred = model.predict(X_val)

# Calculate accuracy of the model
mae = mean_absolute_error(y_val, y_pred)
print("Mean Absolute Error:", mae)

real_y_Ein = model.predict(X_train)
for i in range(len(real_y_Ein)):
    if real_y_Ein[i] < 0:
        real_y_Ein[i] = 0
    if real_y_Ein[i] > 9:
        real_y_Ein[i] = 9
real_Ein = mean_absolute_error(y_train, real_y_Ein)
print("Real Ein:", real_Ein)

test_data = pd.read_csv("test_data.csv", usecols=['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Likes', 'Composer', 'Artist'])
# test_data = pd.read_csv("test.csv", usecols=['Energy', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Tempo', 'Composer', 'Artist'])
column_averages = test_data.iloc[:, :17].mean(skipna=True)
column_medians2 = test_data.iloc[:, :17].median()
test_data.iloc[:, :17] = test_data.iloc[:, :17].fillna(column_averages)
# print(test_data)

X = scaler.transform(X)
y_Ein = model.predict(X)
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
ans = model.predict(test_data)
mean_ans = np.mean(ans)  # Calculate the mean of 'ans'
print("Mean of ans:", mean_ans)
np.set_printoptions(threshold=np.inf)
ans = np.squeeze(ans)
# print(ans)
print(type(ans))

for i in range(len(ans)):
    # ans[i] += (mean_y - mean_ans)
    if ans[i] < 0:
        ans[i] = 0
    if ans[i] > 9:
        ans[i] = 9
    # if ans[i] > 6:
    #     ans[i] = math.floor(ans[i])
    # elif ans[i] < 3:
    #     ans[i] = math.ceil(ans[i])
    # else:
    rounded = int(ans[i] + 0.5) if ans[i] > 0 else int(ans[i] - 0.5)
    ans[i] = rounded

array = [0,0,0,0,0,0,0,0,0,0]
for i in range(len(ans)):
    array[int(ans[i])] = array[int(ans[i])]+1

print(array)

#os.remove("./NNR.csv")
csv_file = "NNR.csv"

# Generate the index values starting from 17170
index_values = range(17170, 17170 + len(ans))

# Write the predicted danceability values to the CSV file
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'Danceability'])
    writer.writerows(zip(index_values, ans))

print("CSV file 'NNR.csv' created successfully.")
