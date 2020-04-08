import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#   Set Pandas printoptions
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

#   Load the data
path = r"C:\Users\Piet\Documents\Learn Programming\Datascience with Python\CSV files to read\goodreads.csv"
dfbooks = pd.read_csv(path)

#   Repair the data: the first row is used as the header.
#   Actually, this happens because there are no headers. The quick way would have been to load the data
#   using header names.

#   First: construct a dataframe containing the one book used as column name in 'dfbooks'.
array_of_column_names = dfbooks.columns.values
list_of_new_names = ['rating', 'rev_count', 'isbn', 'type', 'author_url', 'year',
                                 'genre_url', 'dir', 'rat_count', 'title']
first_row = pd.DataFrame({list_of_new_names[i] : array_of_column_names[i] \
    for i in range(len(array_of_column_names))}, index=[0])
#   Next: replace the header names of dfbooks and concatenate the two dataframes
dfbooks = dfbooks.rename(columns={array_of_column_names[i] : list_of_new_names[i] \
                                  for i in range(len(array_of_column_names))})
dfbooks = pd.concat([first_row, dfbooks], ignore_index=True, sort=False)

#   Cleaning up the data by removing NaN values and changing from type 'object' to type 'int/float' where possible
#   The following are bad rows, let's remove them.
dfbooks = dfbooks[dfbooks.rating.notnull()]     # Keeps the rows where 'rating' has numerical value
dfbooks = dfbooks[dfbooks.year.notnull()]
#   The 'NaN' entries in some columns should be empty strings.
dfbooks.loc[dfbooks.genre_url.isnull(), 'genre_url'] = ""
dfbooks.loc[dfbooks.isbn.isnull(), 'isbn'] = ""
#   Let's try to convert some object types to either float or int.
dfbooks.rating = dfbooks.rating.astype(float)
dfbooks.rev_count = dfbooks.rev_count.astype(int)
dfbooks.year = dfbooks.year.astype(int)
dfbooks.rat_count = dfbooks.rat_count.astype(int)

#   Get the author from author_url and the genre from genre_url. Add columns containing the author and genre
#   of a book.
def get_author(url):
    last_dot_before_name = url.rfind('.')
    author = url[last_dot_before_name + 1:]
    return author

def get_genre(url):
    genres = url.replace('/genres', '')
    return genres
# Add author and genre column using the '.map' method.
dfbooks['author'] = dfbooks.author_url.map(get_author)
dfbooks['genre'] = dfbooks.genre_url.map(get_genre)
#print(dfbooks.shape)


#   Plot the histograms of numerical columns. In some cases we choose logarithmic scale to get
#   an informative plot
#plt.figure(figsize=(12,6))
#plt.subplot(221)
#plt.hist(dfbooks.rating, bins=60)
#plt.xlabel('rating')
#plt.ylabel('frequency')
#plt.title('Rating')
#plt.axvline(x=dfbooks.rating.mean(), color='red')

#plt.subplot(222)
#plt.hist(dfbooks.rat_count, bins=60)
#plt.xlabel('rating count')
#plt.ylabel('frequency')
#plt.title('Rating count')
#plt.yscale('symlog')
#plt.axvline(x=dfbooks.rat_count.mean(), color='red')

#plt.subplot(223)
#plt.hist(dfbooks.year, bins=60)
#plt.xlabel('year')
#plt.ylabel('frequency')
#plt.title('Year of publication')
#plt.yscale('symlog')
#plt.axvline(x=dfbooks.year.mean(), color='red')

#plt.subplot(224)
#plt.hist(dfbooks.rev_count, bins=60)
#plt.xlabel('review count')
#plt.ylabel('frequency')
#plt.yscale('symlog')
#plt.axvline(x=dfbooks.rev_count.mean(), color='red')

#plt.show()

#   Determine the best book in each year.
#for year, subset in dfbooks.groupby('year'):
#    best_book = subset.loc[subset.rating.idxmax()]
#    print(best_book)

#   Analyse the development of genres.

#   Create a list with all genres.
set_of_genres = set(dfbooks.genre)  # extract genres from the dataframe
set_of_splitted_genres = set()
#   Split the subgenres and put them in a list
for genre in set_of_genres:
    splitted_genres = genre.split('|')
    for splitted_genre in splitted_genres:
        splitted_genre_without_slash = splitted_genre[1:]   # ignore the first character, which is a '/'
        set_of_splitted_genres.update(splitted_genre_without_slash)

#   Define a function that tells you whether a book is in a certain genre, based on the string of genres
#   from the book.
def find_genre_in_string(genres, genre_in_this_column=""):
    genre_found = genres.find(genre_in_this_column)
    if genre_found >= 1:
        return 1
    else:
        return 0

#   Append a new column for every genre to dfbooks. The entries are either '1' or '0', depending on whether
#   the book is or isn't in that genre, respectively.
#   For an alternative solution, see the solutions of this tutorial on github. It can be done in two lines!!!
for genre in set_of_splitted_genres:
    def find_genre_given_column(genres):
        return find_genre_in_string(genres, genre_in_this_column=genre)
    dfbooks[genre] = dfbooks.genre.map(find_genre_given_column)

#   Analyse the development of genres: get the five most popular genres (overall) and see how the
#   popularity of those genres developed over the years.
#   Create a sorted dictionary with keys the genres, and values the number of occurences of each genre in goodreads.
dictionary_with_genre_amounts = {genre : dfbooks[genre].sum() for genre in dfbooks.columns.values[13:]}
list_with_genre_amounts_sorted = sorted(dictionary_with_genre_amounts, \
                                              key=dictionary_with_genre_amounts.get, reverse=True)
list_with_10_most_popular_genres = list_with_genre_amounts_sorted[:10]

# Create a list containing the years
list_of_years = sorted(set(dfbooks['year'].values))

plt.figure(figsize=(12,12))
for i in range(1, 10, 1):
    #   Create a list for a given genre with the number of books per year
    popularity_genre = []
    genre = list_with_10_most_popular_genres[i - 1]
    for year, group in dfbooks.groupby('year'):
        popularity_genre_this_year = group[genre].sum()     # See how often a genre was present for each year
        popularity_genre.append(popularity_genre_this_year)
    plt.subplot(3,3,i)
    plt.plot(list_of_years, popularity_genre)
    plt.yscale('symlog')
    plt.title(genre)

plt.show()





