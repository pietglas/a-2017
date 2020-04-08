import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#   Load data
df = pd.read_csv('CSV files to read/noisypopulation.csv')

#   Set variables
#x = df.x.values
#y = df.y.values
#f = df.f.values

#   Randomly sample 60 points out of the 200
#indexes = np.sort(np.random.choice(x.shape[0], size=60, replace=False))
#x_sample = x[indexes]
#y_sample = y[indexes]
#f_sample = f[indexes]
#df_sample = pd.DataFrame(dict(x = x_sample, f = f_sample, y = y_sample))
df_sample = df.sample(n=60)     # another way to obtain a sample of 60

#   Split the data into a training and testing set
train_df, test_df = train_test_split(df_sample, train_size=0.8)
train_x = train_df[['x']]
test_x = test_df[['x']]
train_y = train_df.y
test_y = test_df.y

#   Split training data further into a training and validation set
pretrain_df, valid_df = train_test_split(train_df, train_size=36, test_size=12)
pretrain_x = pretrain_df[['x']]
valid_x = valid_df[['x']]
pretrain_y = pretrain_df.y
valid_y = valid_df.y

#   Apply PolynomialFeatures in order to apply multilinear regression
#   Create function that makes a test and train dictionary with key the degrees d, and values the train/test data after
#   applying PolynomialFeatures(d) to it.
def make_polynomial_features(train_set, test_set, degree):
    degrees = range(degree)
    pretrain_dict = {d : PolynomialFeatures(d).fit_transform(train_set) for d in degrees}
    valid_dict = {d : PolynomialFeatures(d).fit_transform(test_set) for d in degrees}
    return pretrain_dict, valid_dict

#   Make the train and test dictionaries.
degree = 21
pretrain_dict, valid_dict = make_polynomial_features(pretrain_x, valid_x, degree)
#   Create arrays in which we'll store the mean squared errors for all degrees less than 21
error_pretrain = np.empty([1, 21])
error_valid = np.empty([1, 21])

#   Train the models and store the errors for each model
for d in range(degree):
    #   make the model
    regr = linear_model.LinearRegression()
    results = regr.fit(pretrain_dict[d], pretrain_y)
    #   construct linear regression for the training set
    regr_results = regr.predict(pretrain_dict[d])
    #   predict approximation for the test set
    pred = regr.predict(valid_dict[d])
    #   determine and save the mean squared errors
    mse_train = mean_squared_error(pretrain_y, regr_results)
    mse_test = mean_squared_error(valid_y, pred)
    error_pretrain[0, d] = mse_train
    error_valid[0, d] = mse_test

#   Plot the degree against the MSE
degrees = np.arange(21).reshape(1,21)
best_degree = np.argmin(error_valid)     # Find the degree that gives smallest MSE
print(best_degree)
plt.plot(degrees, error_pretrain, '.', color='blue', label='train (in-sample)')
plt.plot(degrees, error_valid, '.', color='green', label='test')
plt.axvline(best_degree, color='r', label='min test error at d=%d'%best_degree, alpha=0.3)
plt.xlabel('degree')
plt.ylabel('MSE')
plt.yscale('log')
plt.show()

#   Retrain the model with the best degree on the whole training set, test on test set
#   First, apply PolynomialFeatures(best_degree) to training and testing set
x_train_transformed = PolynomialFeatures(best_degree).fit_transform(train_x)
x_test_transformed = PolynomialFeatures(best_degree).fit_transform(test_x)
#   Train the model
regr = linear_model.LinearRegression()
results = regr.fit(x_train_transformed, train_y)
#   Make predictions on the testing set
regr_results = regr.predict(x_train_transformed)
pred = regr.predict(x_test_transformed)
#   Calculate the MSE
mse_train = mean_squared_error(train_y, regr_results)
mse_test = mean_squared_error(test_y, pred)
print('Mean squared errors for training and testing: \n', mse_train, mse_test)

