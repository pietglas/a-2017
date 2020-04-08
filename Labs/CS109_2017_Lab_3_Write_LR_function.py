import numpy as np
import pandas as pd
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm

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

#   Plot the regression line against the toy data, using our function to calculate the regression line

#   Toy data
predictors = [1, 2, 3]
responses = [2, 2, 4]

#   Calculate regression parameters and regression line using predifined function
parameters = linear_regression_fit(np.array([[1], [2], [3]]), responses=[2, 2, 4])
parameters_list = [parameters[i][0] for i in range(len(parameters))]
regression_line = regression_line(1, 2, 3, parameters=parameters_list)

#   Calculate regression parameters using sklearn
predictors_m = np.array(predictors).reshape(len(predictors), 1)
responses_m = np.array(responses).reshape(len(responses), 1)
model_skl = linear_model.LinearRegression()
parameters_skl = model_skl.fit(predictors_m, responses_m)
#print('model intercept: ', model_skl.intercept_)
#print('model slope: ', model_skl.coef_[0])

#   Plot
#plt.plot(predictors, responses, '.');
#plt.plot(predictors, regression_line, '-');
#plt.show()





