import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from CS109_2017_Lab_Functions import make_polynomial_features


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

#   Set degree
degree = 21

#   Function that returns optimal regularized Ridge model wrt MSE
def cv_optimize_ridge(X, y, list_of_parameters, n_folds=4):
    est = Ridge()
    parameters = {"alpha" : list_of_parameters}
    gs = GridSearchCV(est, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error')
    gs.fit(X,y)
    return gs

list_of_parameters = [1e-8, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 1e-2, 0.1, 0.0, 1.0]
pretrain_dict, valid_dict = make_polynomial_features(train_x, test_x, degree)
Xtrain = pretrain_dict[20]
Xtest = valid_dict[20]

#   Find best model for a fixed degree
fitmodel = cv_optimize_ridge(Xtrain, train_y)
print(fitmodel.best_estimator_)
print(fitmodel.best_params_)
print(fitmodel.best_score_)

#   Compare the different alphas to their mean squared error
fit_alphas = [d['alpha'] for d in fitmodel.cv_results_['params']]
fit_scores = fitmodel.cv_results_['mean_test_score']
plt.scatter(np.log10(fit_alphas), np.log10(-fit_scores))
plt.xlabel('$\log_{10}(\lambda)$')
plt.ylabel('$-\log_{10}(\mathrm{scores})$')
plt.show()

#   retrain the model with the alpha that gives the best results
best_alpha = fitmodel.best_params_['alpha']
est = Ridge(alpha=best_alpha)
results = est.fit(Xtrain, train_y)
pred = est.predict(Xtest)
mse = mean_squared_error(test_y, pred)
print('The mean squared error for the predicted regression line with alpha = %f is: ' % best_alpha, mse)
print('The regression coefficients are: ', est.intercept_, est.coef_)
