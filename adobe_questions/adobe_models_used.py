# Importing the libraries
import pandas as pd
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split   # Using Skicit-learn to split data into training and testing sets
from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


# First step is to inspect data and perform data cleaning
# Read csv file into a pandas data frame
df1 = pd.read_csv("intern_data.csv", index_col=0)
df2 = pd.read_csv("intern_test.csv")

# Determine if there is any missing value (Note that I didn't normalise the input values because through inspection
# of the minimum and maximum values of each feature array, the range lies between 0 to 1, so there is no need to
print('Checking if there is any missing values in the datasets:')
missing_exist = (df1.isnull().sum())
missing_exist1 = (df2.isnull().sum())

for i in missing_exist:
    if i != 0:
        print("There are missing values in training data. Need further inspection\n")

print("No missing values in the training data. Good to go")

for i in missing_exist1:
    if i != 0:
        print("There are missing values in testing data. Need further inspection\n")

print("No missing values in the testing data. Good to go\n")

# save the column index of intern_test.csv into an array
column_index = df2.get(df2.columns[0])
column_index = array(column_index)

# now we can remove the first column of the intern_test.csv for model predictions
df2 = df2.drop(df2.columns[0], axis=1)

# One-hot encode the data using pandas get_dummies because there are string values
df1 = pd.get_dummies(df1)
df2 = pd.get_dummies(df2)

# get all the feature and store it to X1, notice that all the features are used because after examining feature
# importance using SHAP values, recursive feature elimination (RFE), and Lasso, features like a, h_white, c_yellow etc.
# have low importance, but does not represent noise. Removing them do not necessarily improve the results
X1 = df1.drop('y', axis=1)
Y1 = df1.y
X2 = df2


# Split the data into training and testing sets, splitting the intern_data.csv into 70% training data and 30% testing
# data
train_features, test_features, train_labels, test_labels = train_test_split(X1, Y1,
                                                                            test_size=0.3, random_state=42)

# First one that I've tried is Neural networks


# functions to build the neural network regressor
def build_regressor():
    regressor = Sequential()
    regressor.add(Dense(units=10, input_dim=12))            # input_dim of 12 for the 12 features
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mae', 'accuracy'])
    return regressor


# the optimizer, the number of epochs and batch size were determined using GridSearchCV, with the best possible outcome
regressor = KerasRegressor(build_fn=build_regressor, batch_size=10, epochs=100)
results=regressor.fit(train_features, train_labels)

# predict the y values for the testing data
prediction1 = regressor.predict(test_features)

# measure the error to evaluate the accuracy of our current model. Here, mean absolute error, mean square error, and
# root mean square error were used as there are more suited for regression tasks, unlike other metrics for
# classifications, eg, confusion matrix
errors = abs(prediction1 - test_labels)

# Print out the mean absolute error, mean square error, and root mean square error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print("Mean Square Error:", round(mean_squared_error(test_labels, prediction1), 4))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, prediction1)))

# # Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# print(mape)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
# print(rf.score(X, Y))
print("R2 score:", r2_score(test_labels, prediction1))

# Prediction for the test dataset
predictions2 = regressor.predict(X2)
array_data = np.vstack((column_index, predictions2))
array_data = np.transpose(array_data)

# Second one is Random Forest regressor
# The parameters for the random forest regressor were determined using RandomSearch and GridSearch
rf = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_features=3,
                           max_depth=None, bootstrap=False)

# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)     # Calculate the absolute errors
# measure the error to evaluate the accuracy of our current model. Here, mean absolute error, mean square error, and
# root mean square error were used as there are more suited for regression tasks, unlike other metrics for
# classifications, eg, confusion matrix
errors = abs(prediction1 - test_labels)     # Print out the mean absolute error (mae)

# Print out the mean absolute error, mean square error, and root mean square error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print("Mean Square Error:", round(mean_squared_error(test_labels, prediction1), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(test_labels, prediction1)), 4))

# Calculate the mean absolute percentage error (MAPE) and the R^2 score of the model
mape = np.mean(100 * (errors / test_labels))
print('Mean Absolute Percentage Error:', round(mape, 4))
print("R2 score:", round(r2_score(test_labels, prediction1), 4))  # R^2 (coefficient of determination)
                                                                  # regression score function

# Prediction for the intern_test.csv dataset using the model that was trained
predictions2 = lm.predict(X2)
array_data = np.vstack((column_index, predictions2))   # merge the column index with the predictions
array_data = np.transpose(array_data)                  # transpose it to the right dimension


# 3 Decision Trees Regressor that was used
dt = DecisionTreeRegressor()
dt.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = dt.predict(test_features)     # Calculate the absolute errors

# measure the error to evaluate the accuracy of our current model. Here, mean absolute error, mean square error, and
# root mean square error were used as there are more suited for regression tasks, unlike other metrics for
# classifications, eg, confusion matrix
errors = abs(prediction1 - test_labels)     # Print out the mean absolute error (mae)

# Print out the mean absolute error, mean square error, and root mean square error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print("Mean Square Error:", round(mean_squared_error(test_labels, prediction1), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(test_labels, prediction1)), 4))

# Calculate the mean absolute percentage error (MAPE) and the R^2 score of the model
mape = np.mean(100 * (errors / test_labels))
print('Mean Absolute Percentage Error:', round(mape, 4))
print("R2 score:", round(r2_score(test_labels, prediction1), 4))  # R^2 (coefficient of determination)
                                                                  # regression score function

# Prediction for the intern_test.csv dataset using the model that was trained
predictions2 = lm.predict(X2)
array_data = np.vstack((column_index, predictions2))   # merge the column index with the predictions
array_data = np.transpose(array_data)                  # transpose it to the right dimension

