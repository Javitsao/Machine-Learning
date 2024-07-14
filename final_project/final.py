import pandas as pd
import numpy as np
from sklearn import linear_model

train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Description', 'Composer', 'Artist'])

keywords = ['beethoven', 'Gozzi', 'puccini', 'piano']
train_data['Keywords_Present1'] = train_data['Description'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in keywords)))
# keywords = ['Roll', 'hey', 'Boney']
keywords = ['?', '!', 'crazy', 'Crazy', 'Rock', 'Roll']
train_data['Keywords_Present2'] = train_data['Description'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in keywords)))
# composers = ['Finneas']
# train_data['Composer_Present'] = train_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# train_data['Artist_Present'] = train_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

train_data = train_data.drop('Description', axis=1)
train_data = train_data.drop('Composer', axis=1)
train_data = train_data.drop('Artist', axis=1)

print(train_data)

#train_data.insert(1, 'New_Column', 1)

column_averages = train_data.iloc[:, :17].mean(skipna=True)
train_data.iloc[:, :17] = train_data.iloc[:, :17].fillna(column_averages)
#print(train_data)
X = train_data.iloc[:, 1:17]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column

regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regr.coef_)
test_data = pd.read_csv("test.csv", usecols=['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Description', 'Composer', 'Artist'])
keywords = ['beethoven', 'Gozzi', 'puccini', 'piano']
test_data['Keywords_Present1'] = test_data['Description'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in keywords)))
keywords = ['?', '!', 'crazy', 'Crazy', 'Rock', 'Roll']
test_data['Keywords_Present2'] = test_data['Description'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in keywords)))
# composers = ['Finneas']
# test_data['Composer_Present'] = test_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# test_data['Artist_Present'] = test_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

test_data = test_data.drop('Description', axis=1)
test_data = test_data.drop('Composer', axis=1)
test_data = test_data.drop('Artist', axis=1)


column_averages = test_data.iloc[:, :16].mean(skipna=True)
test_data.iloc[:, :16] = test_data.iloc[:, :16].fillna(column_averages)

print(test_data)
res = np.zeros((17170, 2), dtype=object)  # Set dtype=object for both columns
mae = 0
for i in range(17170):
    a = train_data.iloc[i:i+1, 1:]
    predict = regr.predict(a)
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
    a = test_data.iloc[i:i+1, :]
    predict = regr.predict(a)
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
res_df.to_csv("final.csv", index=False)