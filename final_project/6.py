import pandas as pd
import numpy as np
from sklearn import linear_model

train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Description', 'Composer', 'Artist'])
train_data['Description'] = train_data['Description'].str.lower() 
train_data['Composer'] = train_data['Composer'].str.lower() 
train_data['Artist'] = train_data['Artist'].str.lower() 

# keywords1 = ['beethoven', 'nozzi', 'puccini', 'piano', 'no.']
keywords1 = ['tears', 'lonely', 'broken', 'pain', 'miss', 'sad', 'memory', 'silent', 'whisper', 'rain', 'dark', 'cold', 'shadow', 'empty', 'fragile', 'lost', 'fading', 'apart', 'regret', 'haunted', 'nostalgia', 'aching', 'secrets', 'wounded', 'fall', 'winter', 'still', 'goodbye', 'echo', 'fragments', 'weary', 'solitude', 'fragile', 'bittersweet', 'sorrow', 'teardrop', 'melancholy', 'fade', 'alone', 'drowning', 'broken', 'hurt', 'fragile', 'forgotten', 'ghost', 'broken', 'empty', 'whisper', 'silence', 'shattered', 'lost', 'pain', 'darkness', 'regret', 'shadow', 'aching', 'lonely', 'tear', 'sorrow', 'fragments', 'solitude', 'wounds', 'crying', 'sadness', 'fading', 'heartbreak', 'forgotten', 'secrets', 'echoes', 'tired', 'quiet', 'haunting', 'bruised', 'gloomy', 'grief', 'despair', 'aching', 'yearning', 'desolate', 'farewell', 'bleed', 'remorse', 'bitter', 'hollow', 'aching', 'heartache', 'wither', 'regret', 'shadows', 'forsaken', 'no.']
train_data['Keyword_Present1'] = train_data['Description'].str.count('|'.join(keywords1))
median_count = np.median(train_data['Keyword_Present1'].dropna())
train_data['Keyword_Present1'].fillna(median_count, inplace=True)
# keywords = ['Roll', 'hey', 'Boney']

# keywords2 = ['!', 'crazy', 'Rock', 'roll', 'hey,', 'yeah']
keywords2 = ['love', 'night', 'fire', 'desire', 'dream', 'heart', 'baby', 'party', 'feel', 'move', 'beat', 'dance', 'music', 'tonight', 'body', 'groove', 'floor', 'rhythm', 'touch', 'electric', 'energy', 'up', 'jump', 'shine', 'celebrate', 'high', 'funky', 'wild', 'together', 'passion', 'sweat', 'kiss', 'explosion', 'freedom', 'unite', 'crazy', 'vibe', 'club', 'stars', 'happy', 'magic', 'heat', 'run', 'get', 'let', 'fly', 'pump', 'joy', 'rock', 'shake', 'breathe', 'summer', 'wind', 'adrenaline', 'drum', 'guitar', 'fantasy', 'fantastic', 'fantasia', 'inspire', 'melody', 'sing', 'power', 'scream', 'loud', 'hands', 'clap', 'groovy', 'jive', 'joyful', 'bounce', 'energy', 'beat', 'hip', 'hop', 'soul', 'funk', 'disco', 'electronic', 'party', 'wild', 'move', 'feel', 'tonight', 'celebrate', 'jump', 'rhythm', 'upbeat', 'vibe', 'dance', 'party', 'energy', 'love', 'fire', 'dream', 'groove', 'crazy', 'roll', '!']
train_data['Keyword_Present2'] = train_data['Description'].str.count('|'.join(keywords2))
median_count = np.median(train_data['Keyword_Present2'].dropna())
train_data['Keyword_Present2'].fillna(median_count, inplace=True)
# composers = ['Finneas']
# train_data['Composer_Present'] = train_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# train_data['Artist_Present'] = train_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

train_data = train_data.drop('Description', axis=1)
train_data = train_data.drop('Composer', axis=1)
train_data = train_data.drop('Artist', axis=1)

print(train_data)

#train_data.insert(1, 'New_Column', 1)

column_medians = train_data.iloc[:, :16].median()
train_data.iloc[:, :16] = train_data.iloc[:, :16].fillna(column_medians)
#print(train_data)
X = train_data.iloc[:, 1:16]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column

regr = linear_model.LinearRegression()
regr.fit(X, y)
#print(regr.coef_)
test_data = pd.read_csv("test.csv", usecols=['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Stream', 'Description', 'Composer', 'Artist'])
test_data['Description'] = test_data['Description'].str.lower() 
test_data['Composer'] = test_data['Composer'].str.lower() 
test_data['Artist'] = test_data['Artist'].str.lower() 

# keywords1 = ['beethoven', 'gozzi', 'puccini', 'piano', 'no.']

test_data['Keyword_Present1'] = test_data['Description'].str.count('|'.join(keywords1))
#median_count = np.median(test_data['Keyword_Present1'].dropna())
test_data['Keyword_Present1'].fillna(median_count, inplace=True)
# keywords = ['Roll', 'hey', 'Boney']

#keywords2 = ['!', 'crazy', 'rock', 'roll', 'hey,', 'yeah']

test_data['Keyword_Present2'] = test_data['Description'].str.count('|'.join(keywords2))
median_count = np.median(test_data['Keyword_Present2'].dropna())
test_data['Keyword_Present2'].fillna(median_count, inplace=True)
# composers = ['Finneas']
# test_data['Composer_Present'] = test_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# test_data['Artist_Present'] = test_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

test_data = test_data.drop('Description', axis=1)
test_data = test_data.drop('Composer', axis=1)
test_data = test_data.drop('Artist', axis=1)


column_averages = test_data.iloc[:, :15].mean(skipna=True)
test_data.iloc[:, :15] = test_data.iloc[:, :15].fillna(column_averages)

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
res_df.to_csv("output6.csv", index=False)