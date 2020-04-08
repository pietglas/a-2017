import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from CS109_2017_Lab_Functions import cv_optimize_classif, score_best_classif

#   Load the data
digits = datasets.load_digits()
#print(digits.images.shape)

#   Lets plot some images of digits.
#fig, axes = plt.subplots(10, 10, figsize =(8, 8))
#fig.subplots_adjust(wspace=0.1, hspace=0.1)

#for i, ax in enumerate(axes.flat):
#    ax.imshow(digits.images[i], cmap='binary', interpolation='nearest')
#    ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
#    ax.set_xticks([])
#    ax.set_yticks([])
#plt.show()

#   Reshape into two dimension, so we can work with pandas
d2d = digits.images.reshape(1797, 64)
#   Create a dataframe
df = pd.DataFrame(d2d)
df['target'] = digits.target
#print(df.head())
#   For simplification, we focus in our classification problem only on the numbers 8 and 9. Note that a quicker
#   solution as the one we did would be to use df_bin = df[df.target.isin([8, 9])]. Note that it matters which
#   numbers we take. If we do '0' and '1', we will get an accuracy of 1.0, as those numbers are easy to distinguish.
df_grouped = df.groupby('target')
df_8 = df_grouped.get_group(8)
df_9 = df_grouped.get_group(9)
df_bin = pd.concat([df_8, df_9], axis=0).sample(frac=1).reset_index(drop=True)
#print(df_bin.head())

#   We use LogisticRegression, which is often suitable for classification problems.
#   First, split the data, and scale the predictor data with StandardScalar()
itrain, itest = train_test_split(range(df_bin.shape[0]), train_size=0.6)
set1 = {}
set1['Xtrain'] = StandardScaler().fit_transform(df_bin[list(range(64))].iloc[itrain, :])
set1['Xtest'] = StandardScaler().fit_transform(df_bin[(list(range(64)))].iloc[itest, :])
set1['ytrain'] = df_bin.target.iloc[itrain]==8
set1['ytest'] = df_bin.target.iloc[itest]==8

#   Set regularization parameters for cross validation
paremeters = {'C' : [1e-20, 1e-15, 1e-10, 1e-5, 1e-3, 1e-1, 1, 10, 10000, 100000]}
parameter = 100000

#   Compare irregularized with regularized performance; first do the unregularized case
unreg_model = LogisticRegression()
unreg_model.fit(set1['Xtrain'], set1['ytrain'])
unreg_training_score = unreg_model.score(set1['Xtrain'], set1['ytrain'])
unreg_test_score = unreg_model.score(set1['Xtest'], set1['ytest'])
print('Score on training data: %f' % unreg_training_score)
print('Score on test data: %f' % unreg_test_score)
#   With regularization:
best_model = score_best_classif(LogisticRegression(), set1['Xtrain'], set1['ytrain'],
                                set1['Xtest'], set1['ytest'], parameters=paremeters, n_folds=5)

#   Calculate the confusion matrix
unreg_matrix = confusion_matrix(set1['ytest'], unreg_model.predict(set1['Xtest']))
reg_matrix = confusion_matrix(set1['ytest'], best_model.predict(set1['Xtest']))
print('False positives in the unregularized case: ', unreg_matrix[0, 1])
print('False negatives in the unregularized case: ', unreg_matrix[1, 0])
print('False positives in the regularized case: ', reg_matrix[0, 1])
print('False negatives in the regularized case: ', reg_matrix[1, 0])










