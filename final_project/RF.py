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
from sklearn.model_selection import GridSearchCV
dif = 100
for rs in range(70, 71):
    #train_data = pd.read_csv("train_data.csv", usecols=['Danceability','Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Composer', 'Artist'])
    train_data = pd.read_csv("train_data.csv", usecols=['Danceability', 'Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Energy', 'Tempo', 'Composer', 'Artist'])

    column_averages = train_data.iloc[:, :14].mean(skipna=True)
    train_data = train_data[(train_data['Danceability'] >= 0) & (train_data['Danceability'] <= 9)]
    # print(train_data)
    column_medians = train_data.iloc[:, :14].median()
    train_data.iloc[:, :14] = train_data.iloc[:, :14].fillna(column_medians)


    X = train_data.iloc[:, 1:14]  # Access columns using .iloc
    y = train_data.iloc[:, 0:1]  # Access the first column
    X = X.values
    # print(y)
    y = y.values.ravel()
    # print(type(X))
    # print(type(y))
    # print(y)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=rs)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Create a Random Forest Regression model
    rf_regressor = RandomForestRegressor(random_state=rs)

    # # Define the hyperparameter grid to search over
    # param_grid = {
    #     'n_estimators': [100, 200, 300],  # Number of trees
    #     'max_depth': [5, 10, 15],         # Maximum depth of trees
    #     'min_samples_leaf': [1, 2, 3],    # Minimum samples per leaf
    #     'max_features': ['sqrt', 'log2']  # Maximum features
    # }

    # # Perform grid search with cross-validation
    # grid_search = GridSearchCV(rf_regressor, param_grid, cv=5, scoring='neg_mean_absolute_error')
    # grid_search.fit(X_train, y_train)

    # # Print the best hyperparameters found
    # print("Best Hyperparameters:", grid_search.best_params_)

    rf_regressor = RandomForestRegressor(n_estimators=169, min_samples_leaf=15, max_features=2, random_state=rs)

    # Predict danceability on the validation set
    rf_regressor.fit(X_train, y_train)

    # Predict danceability on the validation set
    y_pred = rf_regressor.predict(X_val)

    # Calculate accuracy of the model
    mae = mean_absolute_error(y_val, y_pred)
    print("Eval:", mae)

    real_y_Ein = rf_regressor.predict(X_train)
    for i in range(len(real_y_Ein)):
        if real_y_Ein[i] < 0:
            real_y_Ein[i] = 0
        if real_y_Ein[i] > 9:
            real_y_Ein[i] = 9
    real_Ein = mean_absolute_error(y_train, real_y_Ein)
    print("Real Ein:", real_Ein)
    if mae - real_Ein < dif:
        dif = mae - real_Ein
        rec = rs
    X = scaler.transform(X)
    y_Ein = rf_regressor.predict(X)
    for i in range(len(y_Ein)):
        if y_Ein[i] < 0:
            y_Ein[i] = 0
        if y_Ein[i] > 9:
            y_Ein[i] = 9
    # for i in range(len(y_Ein)):
    #     y_Ein[i] = round(y_Ein[i])
    Ein = mean_absolute_error(y, y_Ein)
    print("Ein:", Ein)

    #test_data = pd.read_csv("test_data.csv", usecols=['Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo', 'Duration_ms', 'Views', 'Likes', 'Composer', 'Artist'])
    test_data = pd.read_csv("test_data.csv", usecols=['Loudness', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Valence', 'Energy', 'Tempo', 'Composer', 'Artist'])

    column_averages = test_data.iloc[:, :13].mean(skipna=True)
    column_medians2 = test_data.iloc[:, :13].median()
    test_data.iloc[:, :13] = test_data.iloc[:, :13].fillna(column_averages)
    # print(test_data)

    test_data = test_data.values
    test_data = scaler.transform(test_data)
    # y_pred = logreg.predict(test_data)
    ans = rf_regressor.predict(test_data)
    # np.set_printoptions(threshold=np.inf)
    print(ans)

    for i in range(len(ans)):
        if ans[i] < 0:
            ans[i] = 0
        if ans[i] > 9:
            ans[i] = 9
        ans[i] = round(ans[i])

    # err = 0
    # for i in range(17130):
    #     err += abs(y[i] - y_train[i])

    # print(err / 17130)

    # res = np.zeros((6315, 2), dtype=object)  # Set dtype=object for both columns

    # for i in range(6315):
    #     res[i][0] = int(i) + 17170
    #     res[i][1] = float(ans[i])  # Convert the prediction to float type
        # if(res[i][1] > 9):
        #     res[i][1] = 9
        # if(res[i][1] < 0):
        #     res[i][1] = 0
        # res[i][1] = round(res[i][1])

    # res = test(test_data, 0)
    # r, Ein = test(train_data, 1)
    # print("Ein = ", Ein)

    # # Create a DataFrame from the numpy array
    # res_df = pd.DataFrame(res, columns=['id', 'Danceability'])

    # # os.remove("./LR.csv")
    # # Write the DataFrame to a CSV file
    os.remove("./RF.csv")
    csv_file = "RF.csv"

    # Generate the index values starting from 17170
    index_values = range(17170, 17170 + len(ans))

    # Write the predicted danceability values to the CSV file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'Danceability'])
        writer.writerows(zip(index_values, ans))

    print("CSV file 'RF.csv' created successfully.\n")
    print(str(rs) + " " + str(mae - real_Ein) + "\n")
print(str(rec) + " " + str(dif))
