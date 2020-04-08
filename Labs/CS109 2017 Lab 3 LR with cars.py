import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from CS109_2017_Lab_3_Write_LR_function import linear_regression_fit

#   Use sklearn to predict automobile mileage per gallon (mpg) and evaluate those predictions.

#   Load the data
path = r"C:\Users\Piet\Documents\Learn Programming\Datascience with Python\CSV files to read\mtcars.csv"
dfcars = pd.read_csv(path)

#   Split data into a training and test set
traindf, testdf = train_test_split(dfcars, test_size=0.3, random_state=6)
#   Data for simple regression
x_train = traindf[['hp']]
x_test = testdf[['hp']]
y_train = traindf.mpg
y_test = testdf.mpg
#   Data for multiple regression
X_train = traindf[['hp', 'wt']]
X_test = testdf[['hp', 'wt']]

#   Make a simple regression model with sklearn, using hp (horsepower) as a predictor
simple_reg = linear_model.LinearRegression()
results = simple_reg.fit(x_train, y_train)
check_simple_reg = linear_model.LinearRegression()
check_results = check_simple_reg.fit(x_test, y_test)

#   Make a multiple regression model, using horsepower and weight as predictors
mult_reg = linear_model.LinearRegression()
results_mult = mult_reg.fit(X_train, y_train)
check_mult_reg = linear_model.LinearRegression()
check_mult_results = check_mult_reg.fit(X_test, y_test)

#   Make predictions on the test set using the simple trained model
simple_reg_pred = simple_reg.predict(x_test)
#   To compare, find the true regression line with simple regression for the test data
simple_reg_true = check_simple_reg.predict(x_test)
#   Make predictions on test set with multiple trained model
mult_reg_pred = mult_reg.predict(X_test)
mult_reg_true = check_mult_reg.predict(X_test)

#   The coefficients
print('Coefficients for simple regression using own function: \n',
      linear_regression_fit(x_train, responses=y_train))
print('Coefficients for simple regression: \n', simple_reg.coef_[0], simple_reg.intercept_)
print('Coefficients for multiple regression: \n', mult_reg.coef_, mult_reg.intercept_)
#   Mean squared error (also easy to calculate by hand for numpy arrays)
print('Mean Squared error for simple LR prediction: %.2f'
      % mean_squared_error(y_test, simple_reg_pred))
print('Minimum Mean Squared error for simple LR: %.2f'
      % np.mean((y_test - simple_reg_true)**2))
print('Mean Squared error for multiple LR prediction: %.2f'
      % np.mean((y_test - mult_reg_pred)**2))
print('Minimized Mean Squared error: %.2f'
      % np.mean((y_test - mult_reg_true)**2))
#   The coefficient of determination (1 is perfect, 0 is random, negative means worse than random)
print('Coefficient of determination: %.2f'
      % r2_score(testdf.mpg.values.reshape(-1, 1), simple_reg_pred))

#   Plot the outputs for simple regression
plt.plot(x_test, y_test, '.')
plt.plot(x_test, simple_reg_pred, '-', color='red')
plt.plot(x_test, simple_reg_true, '-', color='green')
plt.xlabel('horsepower')
plt.ylabel('miles per gallon')

plt.show()

