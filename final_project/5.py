import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

train_data = pd.read_csv("train.csv", usecols=['Danceability', 'Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Description', 'Composer', 'Artist', 'Album', 'Track'])
# Fill missing values with a placeholder string for each column
train_data['Description'] = train_data['Description'].fillna('')
train_data['Composer'] = train_data['Composer'].fillna('')
train_data['Artist'] = train_data['Artist'].fillna('')
train_data['Album'] = train_data['Album'].fillna('')
train_data['Track'] = train_data['Track'].fillna('')

# Create a TfidfVectorizer object for each column
vectorizer_description = TfidfVectorizer()
vectorizer_composer = TfidfVectorizer()
vectorizer_artist = TfidfVectorizer()
vectorizer_album = TfidfVectorizer()
vectorizer_track = TfidfVectorizer()

# Fit the vectorizers on the respective columns
vectorizer_description.fit(train_data['Description'])
vectorizer_composer.fit(train_data['Composer'])
vectorizer_artist.fit(train_data['Artist'])
vectorizer_album.fit(train_data['Album'])
vectorizer_track.fit(train_data['Track'])

# Transform the columns into TF-IDF vectors
tfidf_vectors_description = vectorizer_description.transform(train_data['Description'])
tfidf_vectors_composer = vectorizer_composer.transform(train_data['Composer'])
tfidf_vectors_artist = vectorizer_artist.transform(train_data['Artist'])
tfidf_vectors_album = vectorizer_album.transform(train_data['Album'])
tfidf_vectors_track = vectorizer_track.transform(train_data['Track'])

# Average the TF-IDF vectors along the rows to obtain numerical representations
PR_description = tfidf_vectors_description.mean(axis=1)
PR_composer = tfidf_vectors_composer.mean(axis=1)
PR_artist = tfidf_vectors_artist.mean(axis=1)
PR_album = tfidf_vectors_album.mean(axis=1)
PR_track = tfidf_vectors_track.mean(axis=1)

# Add the paragraph representations as new columns in the train_data DataFrame
train_data['PR_Description'] = PR_description
train_data['PR_Composer'] = PR_composer
train_data['PR_Artist'] = PR_artist
train_data['PR_Album'] = PR_album
train_data['PR_Track'] = PR_track

# keywords1 = ['beethoven', 'Gozzi', 'puccini', 'piano', 'No.']
# train_data['Keyword_Present1'] = train_data['Description'].str.count('|'.join(keywords1))
# median_count = np.median(train_data['Keyword_Present1'].dropna())
# train_data['Keyword_Present1'].fillna(median_count, inplace=True)
# # keywords = ['Roll', 'hey', 'Boney']
# keywords2 = ['!', 'crazy', 'Crazy', 'Rock', 'Roll', 'hey,', 'yeah']
# train_data['Keyword_Present2'] = train_data['Description'].str.count('|'.join(keywords2))
# median_count = np.median(train_data['Keyword_Present2'].dropna())
# train_data['Keyword_Present2'].fillna(median_count, inplace=True)

# composers = ['Finneas']
# train_data['Composer_Present'] = train_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# train_data['Artist_Present'] = train_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

train_data = train_data.drop('Description', axis=1)
train_data = train_data.drop('Composer', axis=1)
train_data = train_data.drop('Artist', axis=1)
train_data = train_data.drop('Album', axis=1)
train_data = train_data.drop('Track', axis=1)
# train_data = train_data.drop('PR_Description', axis=1)
# train_data = train_data.drop('PR_Composer', axis=1)
# train_data = train_data.drop('PR_Artist', axis=1)
# train_data = train_data.drop('PR_Album', axis=1)
# train_data = train_data.drop('PR_Track', axis=1)

print(train_data)

#train_data.insert(1, 'New_Column', 1)

column_medians = train_data.iloc[:, :19].median()
train_data.iloc[:, :19] = train_data.iloc[:, :19].fillna(column_medians)
#print(train_data)
X = train_data.iloc[:, 1:19]  # Access columns using .iloc
y = train_data.iloc[:, 0:1]  # Access the first column

regr = linear_model.LinearRegression()
regr.fit(X, y)
#print(regr.coef_)
test_data = pd.read_csv("test.csv", usecols=['Energy', 'Key', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Description', 'Composer', 'Artist', 'Album', 'Track'])
# Fill missing values with a placeholder string for each column
test_data['Description'] = test_data['Description'].fillna('')
test_data['Composer'] = test_data['Composer'].fillna('')
test_data['Artist'] = test_data['Artist'].fillna('')
test_data['Album'] = test_data['Album'].fillna('')
test_data['Track'] = test_data['Track'].fillna('')

# Create a TfidfVectorizer object for each column
vectorizer_description = TfidfVectorizer()
vectorizer_composer = TfidfVectorizer()
vectorizer_artist = TfidfVectorizer()
vectorizer_album = TfidfVectorizer()
vectorizer_track = TfidfVectorizer()

# Fit the vectorizers on the respective columns
vectorizer_description.fit(test_data['Description'])
vectorizer_composer.fit(test_data['Composer'])
vectorizer_artist.fit(test_data['Artist'])
vectorizer_album.fit(test_data['Album'])
vectorizer_track.fit(test_data['Track'])

# Transform the columns into TF-IDF vectors
tfidf_vectors_description = vectorizer_description.transform(test_data['Description'])
tfidf_vectors_composer = vectorizer_composer.transform(test_data['Composer'])
tfidf_vectors_artist = vectorizer_artist.transform(test_data['Artist'])
tfidf_vectors_album = vectorizer_album.transform(test_data['Album'])
tfidf_vectors_track = vectorizer_track.transform(test_data['Track'])

# Average the TF-IDF vectors along the rows to obtain numerical representations
PR_description = tfidf_vectors_description.mean(axis=1)
PR_composer = tfidf_vectors_composer.mean(axis=1)
PR_artist = tfidf_vectors_artist.mean(axis=1)
PR_album = tfidf_vectors_album.mean(axis=1)
PR_track = tfidf_vectors_track.mean(axis=1)

# Add the paragraph representations as new columns in the test_data DataFrame
test_data['PR_Description'] = PR_description
test_data['PR_Composer'] = PR_composer
test_data['PR_Artist'] = PR_artist
test_data['PR_Album'] = PR_album
test_data['PR_Track'] = PR_track

# keywords1 = ['beethoven', 'Gozzi', 'puccini', 'piano', 'No.']
# test_data['Keyword_Present1'] = test_data['Description'].str.count('|'.join(keywords1))
# median_count = np.median(test_data['Keyword_Present1'].dropna())
# test_data['Keyword_Present1'].fillna(median_count, inplace=True)
# # keywords = ['Roll', 'hey', 'Boney']
# keywords2 = ['!', 'crazy', 'Crazy', 'Rock', 'Roll', 'hey,', 'yeah']
# test_data['Keyword_Present2'] = test_data['Description'].str.count('|'.join(keywords2))
# median_count = np.median(test_data['Keyword_Present2'].dropna())
# test_data['Keyword_Present2'].fillna(median_count, inplace=True)

# composers = ['Finneas']
# test_data['Composer_Present'] = test_data['Composer'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in composers)))
# artist = ['Sufjan Stevens']
# test_data['Artist_Present'] = test_data['Artist'].fillna('').apply(lambda x: int(any(keyword in str(x) for keyword in artist)))

test_data = test_data.drop('Description', axis=1)
test_data = test_data.drop('Composer', axis=1)
test_data = test_data.drop('Artist', axis=1)
test_data = test_data.drop('Album', axis=1)
test_data = test_data.drop('Track', axis=1)
# test_data = test_data.drop('PR_Description', axis=1)
# test_data = test_data.drop('PR_Composer', axis=1)
# test_data = test_data.drop('PR_Artist', axis=1)
# test_data = test_data.drop('PR_Album', axis=1)
# test_data = test_data.drop('PR_Track', axis=1)


column_averages = test_data.iloc[:, :18].mean(skipna=True)
test_data.iloc[:, :18] = test_data.iloc[:, :18].fillna(column_averages)

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
res_df.to_csv("output5.csv", index=False)