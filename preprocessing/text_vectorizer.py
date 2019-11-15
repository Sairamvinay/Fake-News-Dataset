from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

'''
Parameters: 2 arrays of strings which contain the text of each article within the dataset for train and test
Returns: 3 arrays: the feature vector for train, the feature vector for test, words (the features fitted on train)
'''
def CV(training_text,testing_text):

	
	cv = CountVectorizer()
	cv.fit(training_text)
	
	X_train = cv.transform(training_text)
	X_test = cv.transform(testing_text)
	words = cv.get_feature_names()
	
	X_train = X_train.todense()
	X_test = X_test.todense()

	return X_train,X_test,words

'''
Parameters: 2 arrays of strings which contain the text of each article within the dataset for train and test
Returns: 3 arrays: the feature vector for train, the feature vector for test, words (the features fitted on train)
'''
def tfidf(training_text,testing_text):

	
	tfidf = TfidfVectorizer()
    tfidf.fit(training_text)

	X_train = tfidf.transform(training_text)
	X_test = tfidf.transform(testing_text)
	words = tfidf.get_feature_names()

	X_train = X_train.todense()
	X_test = X_test.todense()
	
	return X_train,X_test,words
    


def word2vec(training_text, testing_text):
	# First download Google's word embeddings
	# Then load the pre-trained model
	model = gensim.models.Word2Vec.load_word2vec_format(
		"GoogleNews-vectors-negative300.bin",
		binary = True
	)

	X_train = [model[word] for word in training_text]
	X_test = [model[word] for word in testing_text]

	return X_train, X_test