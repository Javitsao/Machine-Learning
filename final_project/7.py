import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream'])

column_averages = train_data.iloc[:, :14].mean(skipna=True)
train_data.dropna(subset=['Danceability'], inplace=True)
train_data.iloc[:, :14] = train_data.iloc[:, :14].fillna(column_averages)

X = train_data.iloc[:, 1:14]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
# Create an instance of the Ridge Regression model
# feature_names = X.columns
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y)
# regr = linear_model.LinearRegression()
# regr.fit(X, y)

test_data = pd.read_csv("test.csv", usecols=['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream'])
# column_averages = test_data.iloc[:, :13].mean(skipna=True)
# test_data.iloc[:, :13] = test_data.iloc[:, :13].fillna(column_averages)
column_medians = test_data.iloc[:, :13].median()
test_data.iloc[:, :13] = test_data.iloc[:, :13].fillna(column_averages)
X_test = scaler.transform(test_data)
print(X_test)

# ridge.feature_names_in_ = feature_names

res = np.zeros((17170, 2), dtype=object)  # Set dtype=object for both columns
mae = 0
for i in range(17170):
    a = X_train[i:i+1, 0:]
    predict = ridge.predict(a)
    res[i][0] = int(i)
    res[i][1] = float(predict)  # Convert the prediction to float type
    if(res[i][1] > 9):
        res[i][1] = 9
    if(res[i][1] < 0):
        res[i][1] = 0
    res[i][1] = round(res[i][1])
    mae += abs(res[i][1] - y.iloc[i])

# Create a DataFrame from the numpy array
res_df = pd.DataFrame(res, columns=['id', 'Danceability'])

# Write the DataFrame to a CSV file
# res_df.to_csv("output2.csv", index=False)
print(mae/17170)
#print(regr.coef_)

res = np.zeros((6315, 2), dtype=object)  # Set dtype=object for both columns
for i in range(6315):
    a = X_test[i:i+1, :]
    predict = ridge.predict(a)
    res[i][0] = int(i) + 17170
    res[i][1] = float(predict)  # Convert the prediction to float type
    if(res[i][1] > 9):
        res[i][1] = 9
    if(res[i][1] < 0):
        res[i][1] = 0
    res[i][1] = round(res[i][1])

# Create a DataFrame from the numpy array
res_df = pd.DataFrame(res, columns=['id', 'Danceability'])

# Write the DataFrame to a CSV file
res_df.to_csv("output7.csv", index=False)
