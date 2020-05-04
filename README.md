# Naive-Bayes-Classifier-Algorithm

## Libraries and functions used : 

Pandas - For dataframe and data pre-processing

Numpy - For Arrays (Easy to convert to pandas Dataframe, save as different file types)

Walk (From Os) - To walk through the folder and get filenames, root for each file

join (From Os) - To Join file paths and root which are given by walk

BeautifulSoup - To remove HTML tags from the email body

Stopwords - To remove stopwords from the email body

PorterStemmer - To get the stem form of the words in the email body

Word_tokenize - To tokenize the words in order to put them separately in a dataframe

Sklearn - For train-test split

# Spam Assasin Corpus.

(Corpus)[https://spamassassin.apache.org/old/publiccorpus/]

## Firstly : 
Download the folders from the above link.

## How to run jupyter notebook ? 
```bash
Install Anaconda Navigator or Run Jupyter on the cloud and upload the .ipynb files and the ham and spam folders.
```

## How to pre-process the data ? 

#### Run the “Final_CS 0 - Getting email body and cleaning the message”
This .ipynb file will :
  1. Clean the email body (Stemming, Tokenizing, Removing Stopwords and Punctuations, Removing HTML tags)
	2. Get the top 4000 most occurring words from the whole dataset. 
	3. Assign word id to each of the words and Generate a full matrix 
  Note: A full matrix contains all the zero values. The number of columns in the full matrix in this file is 7670, this is the number of words in the longest email. 
	4. Split the data in 70% and 30% as Train, Test. ie. X_test, X_train, y_test, y_train
	5. Turns the full matrix into sparse matrix which contains the word ID of the words that are in top 4000 word set



## How to gather the probability ? 
####  Run “Final_CS 1 - Loading saved files and training the naive bayes classifier” 


This file will : 
1. Turn the Sparse Matrix from (Final_Cs 0) to a full matrix. So that for each word Id there is an occurence. Since, it will convert the sparse to a full matrix with word ID as the row, there will be zero values. To fix this issue we will use additive smoothing when finding probability (not to have 0 division error).
2. sum_words_all : Sum up the columns to get Occurence of each word  + 1 ←- Additive smoothing
3. P(Word) (Probability of a word occurring) : sum_words_all / (total number of words + 4000) ←- Additive Smoothing, since we added one for each value in the numerator.
4. sum_spam_words : Sum up the columns to get the occurence of each word in spam series + 1 ←- Additive smoothing
5. P(Word | Spam) (Probability of a word occurring given that the email is spam) : sum_spam_words/ (total words in spam + 4000) ←- Additive smoothing, , since we added one for each value in the numerator.
6. sum_ham_words : Sum up the columns to get the occurence of each word in ham series + 1 ←- Additive smoothing
7. P(Word | ham) (Probability of a word occurring given that the email is ham) : sum_ham_words/ (total words in ham + 4000) ←- Additive smoothing, , since we added one for each value in the numerator.


## How to predict test dataset ? 
#### Run Final_CS 2 - Testing and evaluating the naive bayes classifier.

Using dot product, we are able to run the probability (Gathered from train dataset), to calculate the probability of spam and ham for test dataset. 



## How to predict user given email ? 
#### Run Final_CS 3 - Predicting User given email notebook.

1. Calling the predict_user_email() function, asks for input of an email.
2. Upon providing the input, the function passes the email to clean_email_body(email) and returns a filtered email (ie. no stopwords, no html tags, tokenized).
3. The function then creates an empty dataframe and populates the occurrences of word IDs if the words in the user given email occurs in the top 4000 word dataset.
4. A dot product is found using the populated dataframe and the probabilities gathered from the train dataset.
5. The function returns “Spam” and “Non-Spam” based on the combined probabilities of each word from the user given email.


### Sources: 

https://spamassassin.apache.org/old/publiccorpus/
http://www.statsoft.com/textbook/naive-bayes-classifier
https://www.youtube.com/watch?v=z5UQyCESW64
https://www.youtube.com/watch?v=NFd0ZQk5bR4
https://developers.google.com/machine-learning/crash-course/classification/true-false-positive-negative
