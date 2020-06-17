# Importing the libraries
import pandas as pd
import numpy as np
from numpy import array
from sklearn.model_selection import train_test_split   # Using Skicit-learn to split data into training and testing sets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
import matplotlib
import xgboost
import shap

# First step is to inspect data and perform data cleaning
# Read csv file into a pandas data frame
df1 = pd.read_csv("intern_data.csv", index_col=0)
df2 = pd.read_csv("intern_test.csv")

# Determine if there is any missing value. This is done by checking if "isnull().values.any()" is True or False, if True
# means that there is/are missing value(s) in the dataset and vice versa.
# (Note that I didn't normalize the input values because through inspection
# of the minimum and maximum values of each feature array, the range lies between 0 to 1, so there is no need to
# normalize

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
column_index = array(column_index)

# now we can remove the first column of the intern_test.csv for model predictions
df2 = df2.drop(df2.columns[0], axis=1)

# One-hot encode the data using pandas get_dummies because there are string values
df1 = pd.get_dummies(df1)
df2 = pd.get_dummies(df2)

# get all the features and store them to X1, notice that all the features are used because after examining feature
# importance using recursive feature elimination (RFE), Lasso and SHAP values, features like a, h_white, c_yellow etc.
# have low importance, but does not represent noise. Removing them does not necessarily improve the results
X1 = df1.drop('y', axis=1)
Y1 = df1.y
X2 = df2

# First one, is to use recursive feature elimination (RFE) using a linear regressor
print("\n For Recursive Feature Elimination:")
model = LinearRegression()                # Initializing RFE model
rfe = RFE(model, 7)                       # Transforming data using RFE
X_rfe = rfe.fit_transform(X1, Y1)         # Fitting the data to model
model.fit(X_rfe, Y1)
print(rfe.ranking_)

cols = list(X1.columns)
model = LinearRegression()              # Initializing RFE model
rfe = RFE(model, 10)                    # Transforming data using RFE
X_rfe = rfe.fit_transform(X1, Y1)          # Fitting the data to model
temp = pd.Series(rfe.support_, index=cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)


# Second one is by using LassoCV, we try to see which features are important to us
reg = LassoCV()
reg.fit(X1, Y1)
print("\n For Lasso CV:")
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" % reg.score(X1, Y1))
coef = pd.Series(reg.coef_, index = X1.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")


imp_coef = coef.sort_values()
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()


# Third one is using SHAP values - it can show how much each predictor contributes,
# either positively or negatively
print("\n For SHAP values:")
# load JS visualization code to notebook
shap.initjs()

model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X1, label=Y1), 100)
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X1)
shap.summary_plot(shap_values, X1, plot_type="bar")
shap.summary_plot(shap_values, X1)


