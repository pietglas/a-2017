import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from CS109_2017_Lab_Functions import cv_optimize_classif, score_best_classif

#   Load the data
digits = datasets.load_digits()
#print(digits.images.shape)

#   Reshape into two dimension, so we can work with pandas
d2d = digits.images.reshape(1797, 64)
#   Create a dataframe and add the target
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

#   Standardize the data to make it suitable for PCA and LogisticRegression
df_bin_st = StandardScaler().fit_transform(df_bin.loc[:, df_bin.columns != 'target'])
target = df_bin.target.values

#   Apply PCA to reduce the number of features
#   First, check via plotting how many dimension we can discard
pca = PCA().fit(df_bin_st)
#   plot the cumulative sum of the explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Percentage of variance')
plt.show()
#   Now we transform the data. The first two components explain about 35 percent of the variance, so we just
#   stick to those.
pca_digits = PCA(n_components=2)
X_pca = pca_digits.fit_transform(df_bin_st)

#   We use LogisticRegression, which is often suitable for classification problems.
#   Generate indices to split the data into training and testing set
itrain, itest = train_test_split(range(df_bin_st.shape[0]), train_size=0.6)
xtrain = X_pca[itrain]
xtest = X_pca[itest]
ytrain = target[itrain]
ytest = target[itest]

#   Set regularization parameters for cross validation
parameters = {'C' : [1e-20, 1e-15, 1e-10, 1e-5, 1e-3, 1e-1, 1, 10, 10000, 100000]}
#   Apply cross validation to find the best regularization parameter and find the score of the best model
best_model = score_best_classif(LogisticRegression(), xtrain, ytrain, xtest, ytest, parameters=parameters, n_folds=5)

#   Calculate the confusion matrix
reg_matrix = confusion_matrix(ytest, best_model.predict(xtest))
print('False positives in the regularized case: ', reg_matrix[0, 1])
print('False negatives in the regularized case: ', reg_matrix[1, 0])



