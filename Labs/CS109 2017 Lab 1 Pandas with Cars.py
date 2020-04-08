import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#   Load the data with Pandas and show the first five rows
path = r"C:\Users\Piet\Documents\Learn Programming\Datascience with Python\CSV files to read\mtcars.csv"
dfcars = pd.read_csv(path)
print(dfcars.head())
#print(dfcars.shape)
drat_mean = dfcars.drat.mean()
drat_std = dfcars.drat.std()

#   Plot the distribution of 'drat' with Pyplot. Seaborn sets up plotting styles and gives more
#   plotting options.
#sns.set_context("poster")   # makes text bigger
#plt.hist(dfcars.drat.values, bins=20)    # via Numpy, bins sets width of the bars in the histogram
#dfcars.mpg.hist(bins=20, grid=False)    # via Pandas
#plt.xlabel("drt")
#plt.ylabel("Frequency")
#plt.title("Distribution rear axle ratio")
#plt.axvline(x=drat_mean, color='red')   # draws a red vertical line at the mean
#   Add text stating the mean and standard deviation of the distribution
#plt.text(4.25, 5.5, "Mean: " + str(dfcars.drat.mean()))
#plt.text(4.25, 5, "Standard deviation: " +str(drat_std))
#plt.show()

#   GENERAL REMARK: DON'T ITERATE THROUGH PANDAS SERIES/DATAFRAMES AND NUMPY ARRAYS. THIS IS A
#   RELATIVELTY (VERY) SLOW PROCESS, AS NEW LISTS HAVE TO BE CREATED. INSTEAD, USE SLICING/INDEXING, OR
#   OTHER PANDAS/NUMPY METHODS.

#   Make a scatterplot of 'mpg' against 'hp'
plt.plot(dfcars.mpg, dfcars.hp, 'o')    # same as using 'scatter'
plt.xlabel('miles per gallon')
plt.ylabel('horsepower')
plt.savefig('miles per gallon plotted against horsepower.pdf')      # safe the plot as pdf-file
plt.show()








