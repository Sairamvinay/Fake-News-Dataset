from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim
from sklearn.ensemble import IsolationForest
import itertools

from time import time
import numpy as np
# '''
# Parameters: 2 arrays of strings which contain the text of each article within the dataset for train and test
# Returns: 3 arrays: the feature vector for train, the feature vector for test, words (the features fitted on train)
# '''

RNG = np.random.RandomState(42)
MAX_FEATURES = 10000
def CV(training_text,testing_text):
    cv = CountVectorizer(max_features = MAX_FEATURES)
    cv.fit(training_text)

    X_train = cv.transform(training_text)
    X_test = cv.transform(testing_text)
    words = cv.get_feature_names()

    X_train = X_train.todense()
    X_test = X_test.todense()
    return X_train,X_test,words

# '''
# Parameters: 2 arrays of strings which contain the text of each article within the dataset for train and test
# Returns: 3 arrays: the feature vector for train, the feature vector for test, words (the features fitted on train)
# '''

def TFIDF(training_text,testing_text):
    tfidf = TfidfVectorizer(max_features = MAX_FEATURES)
    tfidf.fit(training_text)
    
    X_train = tfidf.transform(training_text)
    X_test = tfidf.transform(testing_text)
    words = tfidf.get_feature_names()

    X_train = X_train.todense()
    X_test = X_test.todense()

    return X_train,X_test,words


def getVector(model,tokens,size = 100):
    vec = np.zeros(size)
    count = 0
    for word in tokens:
        try:

            vec += model[word]
            count += 1.0

        except KeyError:
            continue

    if count != 0:
        vec = vec / count

    return vec


def word2vec(training_text, testing_text):
    modelTrain = gensim.models.KeyedVectors.load(
            "../fake-news/train_word2vec_model.bin")
    modelTest = gensim.models.KeyedVectors.load("../fake-news/test_word2vec_model.bin")

    X_train = [getVector(modelTrain,sent.split(' ')) for sent in training_text]
    X_test =  [getVector(modelTest,sent.split(' ')) for sent in testing_text]
    return np.array(X_train), np.array(X_test)




def outlierDection(Features,Ftype):

    clf_Iso = IsolationForest(random_state=RNG,n_jobs = -1)
    clf_Iso.fit(Features)
    y_Iso_Forest = clf_Iso.predict(Features)
    result = np.where(y_Iso_Forest == -1)
    result = list(itertools.chain.from_iterable(result))
    print(np.shape(result))
    print(np.shape(y_Iso_Forest))
    percentOutlier = 100.00 *  np.shape(result)[0]/np.shape(y_Iso_Forest)[0]
    print("the percentage of outliers in ",Ftype," is: ",percentOutlier,"%")
    return result,percentOutlier




