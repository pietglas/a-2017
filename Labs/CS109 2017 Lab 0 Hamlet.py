#   Analysis of Shakespeare's Hamlet
import matplotlib.pyplot as plt

#   In the following code we read the lines from Shakespeare's Hamlet, contained in a text file.
#   We create another file and copy the text to it.
#file = open("hamlet.txt", "r")
#hamlet_text = open("hamlettext.txt", "r+")      # 'r+' stand for read+write
#for line in file:
#    hamlet_text.write(line)
#file.close()
#hamlet_text.close()

#   Create a list contained all the words of Hamlet, which we'll analyse later.
file = open("hamlet.txt", "r")
list_of_words_hamlet = []       # list to be filled with the words in Hamlet
for line in file:
    character_list = []     # empty list that will be filled with characters till they form a word
    for character in line:
        # if the character is not a whitespace or punctuation signs, it is added to the currently formed
        # word.
        if character != " " and character != "." and character != "," and character != "?" and \
            character != "!" and character != "]" and character != "[" and character != "-" and \
            character != ":" and character != ";":
            character_list.append(character)
        else:           # add a new word to list_of_words_hamlet
            new_word = ""
            for character_in_list in character_list:
                new_word += character_in_list
            # We want to make sure that we aren't adding the empty word to the list. If the new_word is
            # empty, continue the loop with the next character.
            if new_word == "":
                continue
            # Decapitalize the word so we can properly count it's occurrence.
            new_word_decapitalized = new_word.lower()
            # Add the word to the list
            list_of_words_hamlet.append(new_word_decapitalized)
            character_list = []
file.close()

#   Print the total number of words in Hamlet
print("There are " + str(len(list_of_words_hamlet)) + " words in Hamlet.")
#   Print the number of unique words in Hamlet
set_of_unique_words_hamlet = set(list_of_words_hamlet)
print("There are " + str(len(set_of_unique_words_hamlet)) + " unique words in Hamlet.")
#   Create a dictionary with the number of occurrences of each unique word, with the unique words
#   as keys.
dict_of_word_occurrences = {word : list_of_words_hamlet.count(word) for word in set_of_unique_words_hamlet}
#   Create a list with the 100 most occurring words in Hamlet, where the first word is the most occurring,
#   the second the second-most, etc.
list_of_unique_words_hamlet_sorted = \
    sorted(dict_of_word_occurrences, key=dict_of_word_occurrences.get, reverse=True)
list_100_most_occurring_words = [list_of_unique_words_hamlet_sorted[i] for i in range(100)]
#for word in list_100_most_occurring_words:
#    print("The word " + word + " occurs " + str(dict_of_word_occurrences[word]) + " times.")

#   Plot the distribution of the 10 most occurring words
topfreq = list_100_most_occurring_words[:10]
plt.bar(topfreq, [dict_of_word_occurrences[word] for word in topfreq])
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.show()









