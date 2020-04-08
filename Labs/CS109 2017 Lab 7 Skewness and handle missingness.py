import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import matplotlib.pyplot as plt
from CS109_2017_Lab_Functions import cv_optimize_classif, score_best_classif

#   We examine several methods to deal with missing values.
#   Question: can we apply regression (either logistic or linear) with categorical predictor variables?

#   Set print options
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#   Load the data
df = pd.read_csv('CSV files to read/gssdata4.csv')
#print(df[0:20])
#print(df.shape)
#print(df.dtypes)

#   Look for asymmetry in the data: only about 6% of the people is in poor health
poorhealth = np.where(df['health'] == 'poor', 1, 0)
excellenthealth = np.where(df['health'] == 'excellent', 1, 0)
fairhealth = np.where(df['health'] == 'fair', 1, 0)
goodhealth = np.where(df['health'] == 'good', 1, 0)
df['poorhealth'] = poorhealth
df['fairhealth'] = fairhealth
df['goodhealth'] = goodhealth
df['excellenthealth'] = excellenthealth
print('%f of the people are in poor health' %(poorhealth.sum()/df.shape[0]))
print('%f of the people are in fair health' %(fairhealth.sum()/df.shape[0]))
print('%f of the people are in good health' %(goodhealth.sum()/df.shape[0]))
print('%f of the people are in excellent health' %(excellenthealth.sum()/df.shape[0]))

#   Convert string into numeric data. Pandas has an inbuilt method for this, get_dummies
dummy_vars = pd.get_dummies(df[['sex', 'sexornt', 'partyid', 'race']])
df = df.join(dummy_vars)
print(df.shape)
print(df.head())
print('From %f of the people their sexual orientation is unknown' % (df['sexornt_dont know'].sum()/df.shape[0]))  # 2%

#######################

#   First method to handle missingness: remove all rows with missing values.
df_full = df.dropna(how = 'any')      # 'any' says that we remove a row already if just one value is missing
print(df_full.shape)
#print(df_full.poorhealth.sum())      # This leaves 963 people, only 16 of which are in poor health!

#   Split the data
#itrain, itest = train_test_split(range(df_full.shape[0]), train_size=0.5)
#   Restrict the number of predictors
#df_t_full = StandardScaler().fit_transform(df_full[['age', 'educ', 'partyid_dem',
#                                                    'partyid_rep', 'income']])
#df_tt_full = df_full[['poorhealth', 'age', 'educ', 'partyid_dem', 'partyid_rep', 'income']]
#print(df_tt_full.poorhealth.sum())     # doesn't affect number of poor people further
#Xtrain = df_t_full[itrain]
#Xtest = df_t_full[itest]
#ytrain = df_full['poorhealth'].iloc[itrain]
#ytest = df_full['poorhealth'].iloc[itest]

#   Regularize and apply LogisticRegression with the best parameter
#clf = LogisticRegressionCV()
#clf.fit(Xtrain, ytrain)
#print(clf.score(Xtest, ytest))

#   reproduce the confusion matrix. We get about 8 false negatives and no true positives. So that means there are
#   8 sick people in the testing set, non of whom was predicted to be sick.
#print(confusion_matrix(ytest, clf.predict(Xtest)))
#yhats = clf.predict_proba(Xtest)
#hist = plt.hist(yhats[:, 1])
#plt.show()

#   Compute the roc curve of the model, gives for various thresholds the rate of true positives out of the positives,
#   against the rate of false positives out of the negatives (standard threshold for a positive is 0.5)
#fpr, tpr, thresholds = roc_curve(ytest, yhats[:, 1])
#ax = plt.gca()
#ax.plot(fpr, tpr, '.-')
#ax.plot([0, 1], [0, 1], 'k--')
#ax.set_xlabel('False positive rate')
#ax.set_ylabel('True positive rate')
#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([0.0, 1.05])

#plt.show()

############################

#   Second method to handle missing values: input the mean or median. Median seems better, since we are talking about
#   income.
#df_med = df.copy()
#df_med['income'] = df['income'].fillna(df_full['income'].median())

#itrain, itest = train_test_split(range(df_full.shape[0]), train_size=0.5)
#   Scale the data, restrict number of predictions
#df_med_r = StandardScaler().fit_transform(df_med[['age', 'educ', 'partyid_dem',
#                                                    'partyid_rep', 'income']])
#Xtrain = df_med_r[itrain]
#Xtest = df_med_r[itest]
#ytrain = df_med['poorhealth'].iloc[itrain]
#ytest = df_med['poorhealth'].iloc[itest]

#   Regularize and apply LogisticRegression with the best parameter
#clf = LogisticRegressionCV()
#clf.fit(Xtrain, ytrain)
#print(clf.score(Xtest, ytest))

#   Compute the roc curve of the model, gives for various thresholds the rate of true positives out of the positives,
#   against the rate of false positives out of the negatives (standard threshold for a positive is 0.5)
#yhats = clf.predict_proba(Xtest)
#fpr, tpr, thresholds = roc_curve(ytest, yhats[:, 1])
#ax = plt.gca()
#ax.plot(fpr, tpr, '.-')
#ax.plot([0, 1], [0, 1], 'k--')
#ax.set_xlabel('False positive rate')
#ax.set_ylabel('True positive rate')
#ax.set_xlim([0.0, 1.0])
#ax.set_ylim([0.0, 1.05])

#plt.show()

############################

#   Third method: use a model (for instance, linear regression) to predict the missing values.
df_lr = df.copy()

#   For scaling the data, using StandardScaler for normally distributed data. Otherwise, use MinMaxScaler
st_scaler = StandardScaler()
#mm_scaler = MinMaxScaler()
df_lr[['age', 'educ']] = st_scaler.fit_transform(df_lr[['age', 'educ']])
#   Divide the data into two dataframes for which the income is either known or unknown
df_nan = df_lr[df_lr.income.isnull()]
df_nnan = df_lr[df_lr.income.notnull()]
#print(df_nan)

#   Train a linear regression model on the predictors for which we know the income
X_pred = df_nan[['age', 'educ', 'sex_female', 'partyid_dem', 'partyid_rep']]
X_train = df_nnan[['age', 'educ', 'sex_female', 'partyid_dem', 'partyid_rep']]
y = df_nnan['income']
linreg = LinearRegression()
linreg.fit(X_train, y)
#   Predict the missing values
pred = linreg.predict(X_pred)
#print(pred)

#   Replace the missing values by the predicted values. In the lab, random noise was added based on the error of
#   the linear regression model.
pred_series = pd.Series(np.array(pred), index=df_nan.index)
#print(pred_series)
df_lr.income[df_lr.income.isnull()] = pred_series
#df_lr['income'] = df_lr['income'].fillna(pred_series)      # equivalent to previous line, but without warning
#print(df.income)
#print(df[df.income.isnull()])      # Check whether all missing values are replaced

#   Split the data into training and test set
itrain, itest = train_test_split(range(df_lr.shape[0]), train_size=0.5)
#   Scale the data, restrict number of predictions
Xtrain = df_lr[['age', 'educ', 'sex_female', 'partyid_dem', 'partyid_rep', 'income']].iloc[itrain]
Xtest = df_lr[['age', 'educ', 'sex_female', 'partyid_dem', 'partyid_rep', 'income']].iloc[itest]
ytrain = df_lr['poorhealth'].iloc[itrain]
ytest = df_lr['poorhealth'].iloc[itest]

#   Regularize and apply LogisticRegression with the best parameter
clf = LogisticRegressionCV()
clf.fit(Xtrain, ytrain)
print(clf.score(Xtest, ytest))

#   reproduce the confusion matrix. We still falsely predict all sick people to be healthy with the standard threshold
#   of 0.5.
plt.subplot(1, 2, 1)
print(confusion_matrix(ytest, clf.predict(Xtest)))
yhats = clf.predict_proba(Xtest)
hist = plt.hist(yhats[:, 1])


#   Compute the roc curve of the model, gives for various thresholds the rate of true positives out of the positives,
#   against the rate of false positives out of the negatives (standard threshold for a positive is 0.5)
#yhats = clf.predict_proba(Xtest)
plt.subplot(1, 2, 2)
fpr, tpr, thresholds = roc_curve(ytest, yhats[:, 1])
ax = plt.gca()
ax.plot(fpr, tpr, '.-')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False positive rate')
ax.set_ylabel('True positive rate')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])

plt.show()




