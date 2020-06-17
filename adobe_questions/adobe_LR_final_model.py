# Importing the libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error


# First step is to inspect data and perform data cleaning
# Read csv file into a pandas data frame
df1 = pd.read_csv("intern_data.csv", index_col=0)
df2 = pd.read_csv("intern_test.csv")

# Determine if there is any missing value. This is done by checking if "isnull().values.any()" is True or False, if True
# means that there is/are missing value(s) in the dataset and vice versa.(Note that the input values was not normalized
# because through inspection, the minimum and maximum values of each feature array, the range lies between 0 to 1,
# so there is no need to normalize

print('Checking if there is any missing values in the datasets:')
if df1.isnull().values.any():
    print("There are missing values in training data. Need further inspection")
else:
    print("No missing values in the intern_data.csv file. Good to go!")

if df2.isnull().values.any():
    print("There are missing values in training data. Need further inspection\n")
else:
    print("No missing values in the intern_test.csv file. Good to go!\n")


# save the column index of intern_test.csv into an array
column_index = df2.get(df2.columns[0])

# now we can remove the first column of the intern_test.csv for model predictions
df2 = df2.drop(df2.columns[0], axis=1)


# One-hot encode the data using pandas get_dummies because there are string values
df1 = pd.get_dummies(df1); df2 = pd.get_dummies(df2)

# Using recursive feature elimination (RFE), LassoCV and SHAP values, the feature selection was performed to determine
# the important features. This is done to reduce over-fitting, improve the accuracy and improve the training time
X1 = df1.drop(['a', 'd', 'h_white', 'c_yellow', 'c_green', 'y'], axis=1)
Y1 = df1.y
X2 = df2.drop(['a', 'd', 'h_white', 'c_yellow', 'c_green'], axis=1)

# Split the data into training and testing sets, splitting the intern_data.csv into 70% training data and 30% testing
# data
train_features, test_features, train_labels, test_labels = train_test_split(X1, Y1,
                                                                            test_size=0.3, random_state=42)


# Function to create the linear regression model, Linear regression was used after trying out neural networks, random
# forest regression and decision trees. It was chosen because it is simple yet produces high accuracy. Neural network
# can achieve similar accuracy as linear regression, but I've decided to go with linear regression due to its low
# computational consumption
lm = linear_model.LinearRegression()
model = lm.fit(train_features, train_labels)

# To make sure the model isn't over-fitting, cross validation score was used to evaluate the estimator's performance
print('Cross validation score is:', cross_val_score(lm, test_features, test_labels, cv=5))

# predict the y values for the testing data
prediction1 = model.predict(test_features)

# measure the error to evaluate the accuracy of our current model.
errors = abs(prediction1 - test_labels)     # Print out the mean absolute error (mae)

# Print out the mean absolute error, mean square error, and root mean square error
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print("Mean Square Error:", round(mean_squared_error(test_labels, prediction1), 4))
print('Root Mean Squared Error:', round(np.sqrt(metrics.mean_squared_error(test_labels, prediction1)), 4))

# Calculate the mean absolute percentage error (MAPE) and the R^2 score of the model
mape = np.mean(100 * (errors / test_labels))
print('Mean Absolute Percentage Error:', round(mape, 4))
print("R2 score:", round(r2_score(test_labels, prediction1), 4))  # R^2 regression score function


# Prediction for the intern_test.csv dataset using the model that was trained
predictions2 = model.predict(X2)
array_data = np.vstack((column_index, predictions2))   # merge the column index with the predictions
array_data = np.transpose(array_data)                  # transpose it to the right dimension

# export the array into a csv file
pd.DataFrame(array_data, columns=['index, i', 'predictions, y']).to_csv("intern_predicted.csv", index=None)
