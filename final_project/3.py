import pandas as pd
import numpy as np
from sklearn import linear_model
import os

def test(data, v):
    if(v):
        Ein = 0
        X = data.iloc[:, 1:14]
        y = data.iloc[:, 0:1]
    else:
        X = data
    res = np.zeros((X.shape[0], 2), dtype=object)  # Set dtype=object for both columns
    for i in range(X.shape[0]):
        a = X.iloc[i:i+1, :]
        predict = regr.predict(a)
        res[i][0] = int(i) + 17170
        res[i][1] = float(predict)  # Convert the prediction to float type
        if(res[i][1] > 9):
            res[i][1] = 9
        if(res[i][1] < 0):
            res[i][1] = 0
        res[i][1] = round(res[i][1])
        
        if(v):
            Ein += abs(res[i][1] - y.iloc[i])
    if(v):
        Ein /= X.shape[0]
        return res, Ein
    else:
        return res

train_data = pd.read_csv("train_data.csv", usecols=['Danceability', 'Energy', 'Loudness', 'Speechiness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Composer', 'Artist'])

column_averages = train_data.iloc[:, :14].mean(skipna=True)
train_data = train_data[(train_data['Danceability'] >= 0) & (train_data['Danceability'] <= 9)]
# print(train_data)
column_medians = train_data.iloc[:, :14].median()
train_data.iloc[:, :14] = train_data.iloc[:, :14].fillna(column_medians)


X = train_data.iloc[:, 1:14]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column

regr = linear_model.LinearRegression()
regr.fit(X, y)

test_data = pd.read_csv("test_data.csv", usecols=['Energy', 'Loudness', 'Speechiness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Composer', 'Artist'])
column_averages = test_data.iloc[:, :13].mean(skipna=True)
column_medians2 = test_data.iloc[:, :13].median()
test_data.iloc[:, :13] = test_data.iloc[:, :13].fillna(column_averages)
# print(test_data)

res = np.zeros((6315, 2), dtype=object)  # Set dtype=object for both columns

for i in range(6315):
    a = test_data.iloc[i:i+1, :]
    predict = regr.predict(a)
    res[i][0] = int(i) + 17170
    res[i][1] = float(predict)  # Convert the prediction to float type
    if(res[i][1] > 9):
        res[i][1] = 9
    if(res[i][1] < 0):
        res[i][1] = 0
    res[i][1] = round(res[i][1])

res = test(test_data, 0)
r, Ein = test(train_data, 1)
print("Ein = ", Ein)

# Create a DataFrame from the numpy array
res_df = pd.DataFrame(res, columns=['id', 'Danceability'])

os.remove("./output3.csv")
# Write the DataFrame to a CSV file
res_df.to_csv("output3.csv", index=False)
