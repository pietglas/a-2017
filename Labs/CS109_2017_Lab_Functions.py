import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#   Create function that makes a test and train dictionary with key the degrees d, and values the train/test data after
#   applying PolynomialFeatures(d) to it.
def make_polynomial_features(train_set, test_set, degree):
    degrees = range(degree)
    pretrain_dict = {d : PolynomialFeatures(d).fit_transform(train_set) for d in degrees}
    valid_dict = {d : PolynomialFeatures(d).fit_transform(test_set) for d in degrees}
    return pretrain_dict, valid_dict

#   Function that returns optimal regularized Ridge model wrt MSE
def cv_optimize_ridge(X, y, list_of_parameters, n_folds=4):
    est = Ridge()
    parameters = {"alpha" : list_of_parameters}
    gs = GridSearchCV(est, param_grid=parameters, cv=n_folds, scoring='neg_mean_squared_error')
    gs.fit(X,y)
    return gs

#   Function that finds the optimal regularization parameter for classifiers using cross-validation. It allows
#   the choice of a performance metric (scoring). For LogisticRegression, we don't need scoring input since
#   it has an inbuilt measure, namely accuracy.
def cv_optimize_classif(clf, Xtrain, ytrain, parameters, n_folds=4, scoring=None):
    if not scoring:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, scoring=scoring)
    gs.fit(Xtrain, ytrain)
    return gs

#   Function that uses cv_optimize_classif to give the score. Can also return for
def score_best_classif(clf, Xtrain, ytrain, Xtest, ytest, parameters, n_folds=4, scoring=None):
    gs = cv_optimize_classif(clf, Xtrain, ytrain, parameters, n_folds=4, scoring=None)
    #   get the best paramater
    best_parameter = 0
    for x in gs.best_params_:
        best_parameter = gs.best_params_[x]
    #   Determine and print the scores
    training_score = gs.score(Xtrain, ytrain)
    test_score = gs.score(Xtest, ytest)
    print('Best parameter: %0.2f' % best_parameter)
    print('Score on training data: %f' % training_score)
    print('Score on test data: %f' % test_score)
    return gs

#   Calculate the parameters for linear regression
def linear_regression_fit(*predictors, responses = np.array([])):
    """
    Calculates the linear regression parameters by solving the normal equation, in case the normal solution exists.

    :param predictors: 2-dimensional numpy array, with each row corresponding to a single observation of the predictors.
    In case of 1 predictor variable, integer input is allowed.
    :param responses: numpy column vector with the observed responses
    :return: numpy column array with the linear regression parameters
    """
    #   Check whether predictors input is of integer type, useful for single predictor variable. If True,
    #   collect the data into a numpy column vector
    predictors = list(predictors)   # change tuple to list, since tuples are immutable
    if isinstance(predictors[0], int):
        for i in range(len(predictors)):
            predictors[i] = [predictors[i]]
        X = np.array(predictors)
    else:
        X = predictors[0]
    print(X)
    print(responses)
    n = len(predictors)
    m = len(responses)

    #   Create the m x (n + 1) matrix X containing the m observations of the n predictors.
    vector_ones = np.ones((len(X),1))
    X = np.concatenate([vector_ones, X], axis=1)

    #   Calculate the transpose of X.
    X_transposed = np.transpose(X)
    print(X_transposed)
    #   Calculate the inverse of the product of the transpose of X with X, if it exists
    X_tX = np.dot(X_transposed, X)
    print(X_tX)
    try:
        X_tX_inverted = np.linalg.inv(X_tX)
    except np.linalg.LinAlgError:
        print("The matrix XX_t does not have an inverse.")
        return 0

    #   Calculate the transpose of the responses
    y_response = np.array(responses)
    y = y_response.reshape(len(responses), 1)

    #   Calculate the solution to the normal equation (throw an exception if XX_transposed is not invertible).
    beta = np.dot(X_tX_inverted, np.dot(X_transposed, y))

    return beta

#   Calculates the regression line from the parameters calculated by linear_regression_fit
def regression_line(*predictors, parameters=[]):
    #   Check whether predictors input is of integer type.
    predictors = list(predictors)  # change tuple to list, since tuples are immutable
    for i in range(len(predictors)):
        if isinstance(predictors[i], int):
            predictors[i] = [predictors[i]]

    #   Add 1 to every list of predictors for convenient calculation of the regression line
    for i in range(len(predictors)):
        predictors[i].insert(0, 1)

    #   Calculate the response as a linear function of the predictors with coefficients from parameters
    y = [sum(parameters[j] * predictors[i][j] for j in range(len(parameters)))
                             for i in range(len(predictors))]

    return y

