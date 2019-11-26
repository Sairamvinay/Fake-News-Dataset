#!/usr/bin/env python
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from plotly.offline import iplot
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

TRAINFILEPATH = "fake-news/train.csv"
TESTFILEPATH = "fake-news/test.csv"

def read_files(PATH,nolabel = False):
    # preprocess and title data
    names = []
    if nolabel == True:
        names = ["id","title","author","text"]

    else:
        names = ["id","title","author","text","label"]
    df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
    # drop NAN values and shuffle data
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# count vectorizerize words, transform corpus to bag of words
# obtrain and sort word frequency in the list
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# counter vectorizerize words, transform corpus to bag of words
# obtain the bigram words frequency and sort them
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


def main():
	# Overall Unigram
	dfTrain = read_files(TRAINFILEPATH,nolabel = False)
	dfTrain['text'].values.astype('U')
	dfTest = read_files(TESTFILEPATH,nolabel = True)
	dfTest['text'].values.astype('U')
    # extract top 20 frequent words as common words
	common_words = get_top_n_words(dfTrain['text'].values.astype('U'), 20)
	for word, freq in common_words:
		print(word, freq)
    # puts words and corresponding counts into approriate data frame
	df1 = pd.DataFrame(common_words, columns = ['text' , 'count'])

    # bar graph the overall Unigram
	objects = (df1['text'])
	y_pos = np.arange(len(objects))
	count = df1['count']
	plt.figure(figsize=(10, 5))
	plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title('Unigram')
	plt.show()


	# Real News Unigram, label and frequency
	dfTrainReal = dfTrain[dfTrain['label']==0]
	common_words = get_top_n_words(dfTrainReal['text'].values.astype('U'), 20)
	for word, freq in common_words:
		print(word, freq)
	df2 = pd.DataFrame(common_words, columns = ['text' , 'count'])

    # Bar graph the real news unigram and corresponding frequency
	objects = (df2['text'])
	y_pos = np.arange(len(objects))
	count = df2['count']
	plt.figure(figsize=(10, 5))
	plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title('Real News Unigram')
	plt.show()


	# False News Unigram, label and frequency
	dfTrainFalse = dfTrain[dfTrain['label']==1]
	common_words = get_top_n_words(dfTrainFalse['text'].values.astype('U'), 20)
	for word, freq in common_words:
		print(word, freq)
	df3 = pd.DataFrame(common_words, columns = ['text' , 'count'])

    # Bar graph the real news unigram and corresponding frequency
	objects = (df3['text'])
	y_pos = np.arange(len(objects))
	count = df3['count']
	plt.figure(figsize=(10, 5))
	plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title('False News Unigram')
	plt.show()


	# Real News Bigram, label and frequency
	dfTrainFalse = dfTrain[dfTrain['label']==0]
	common_words = get_top_n_bigram(dfTrainFalse['text'].values.astype('U'), 20)
	for word, freq in common_words:
		print(word, freq)
	df4 = pd.DataFrame(common_words, columns = ['text' , 'count'])

    # Bar graph the real news biagram and corresponding frequency
	objects = (df4['text'])
	y_pos = np.arange(len(objects))
	count = df4['count']
	plt.figure(figsize=(25, 5))
	plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title('Real News Bigram')
	plt.show()

	# False News Bigram, label and frequency
	dfTrainFalse = dfTrain[dfTrain['label']==1]
	common_words = get_top_n_bigram(dfTrainFalse['text'].values.astype('U'), 20)
	for word, freq in common_words:
		print(word, freq)
	df5 = pd.DataFrame(common_words, columns = ['text' , 'count'])

    # Bar graph the false news biagram and corresponding frequency
	objects = (df5['text'])
	y_pos = np.arange(len(objects))
	count = df5['count']
	plt.figure(figsize=(25, 5))
	plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
	plt.xticks(y_pos, objects)
	plt.ylabel('Count')
	plt.title('False News Bigram')
	plt.show()
