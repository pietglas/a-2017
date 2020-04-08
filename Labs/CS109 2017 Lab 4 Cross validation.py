import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#   Load data
df = pd.read_csv('CSV files to read/noisypopulation.csv')

#   Sample out sixty data points
df_sample = df.sample(n=60)     # another way to obtain a sample of 60
indices = pd.Series(range(60))
df_sample = df_sample.set_index(indices)

#   Split the data into a training and testing set
train_df, test_df = train_test_split(df_sample, train_size=0.8)
train_x = train_df[['x']]
test_x = test_df[['x']]
train_y = train_df.y
test_y = test_df.y

#   Shuffle the data and split it into 4-folds
kfold = KFold(n_splits=4, shuffle=True)     # shuffle the data to make the splits random
kfold.split(train_df)           # generates train - split indices

#   Create function that makes a test and train dictionary with key the degrees d, and values the train/test data after
#   applying PolynomialFeatures(d) to it.
def make_polynomial_features(train_set, test_set, degree):
    degrees = range(degree)
    pretrain_dict = {d : PolynomialFeatures(d).fit_transform(train_set) for d in degrees}
    valid_dict = {d : PolynomialFeatures(d).fit_transform(test_set) for d in degrees}
    return pretrain_dict, valid_dict

#   Apply k-fold cross validation with k = 4.
#   Create arrays in which we'll store the mean squared errors for all degrees less than 21
error_valid = np.empty([4, 21])

#   Loop over the different folds
second_index = 0
for pretrain_index, valid_index in kfold.split((train_df)):
    #   Set variables
    trainfold_x = train_x.iloc[pretrain_index]
    valid_x = train_x.iloc[valid_index]
    trainfold_y = train_y.iloc[pretrain_index]
    valid_y = train_y.iloc[valid_index]

    #   Make the train and test dictionaries, containing PolynomialFeatures of the predictor
    degree = 21
    pretrain_dict, valid_dict = make_polynomial_features(trainfold_x, valid_x, degree)

    #   Loop over the degree d to fit the models, make predictions and calculate the MSE. Note that GridSeachCV
    #   can do this for us (see next exercise).
    for d in range(degree):
        regr = linear_model.LinearRegression()
        results = regr.fit(pretrain_dict[d], trainfold_y)
        pred = regr.predict(valid_dict[d])
        mse = mean_squared_error(valid_y, pred)
        error_valid[second_index, d] = mse
    second_index += 1
    print(second_index)

print(error_valid)

#   Determine the average error and the degree giving the lowest average error
average_error = np.mean(error_valid, axis=0)
best_degree = np.argmin(average_error)

print("The best degree is: ", best_degree)

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
print('Mean squared errors for training and testing, respectively: \n', mse_train, mse_test)



