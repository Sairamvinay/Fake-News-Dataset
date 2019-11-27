from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim

from time import time
import numpy as np

MAX_FEATURES = 10000


# Given samples of training text, for each sample convert it into numerical
# representations (vectors) using CV
def CV(training_text):
    cv = CountVectorizer(max_features = MAX_FEATURES)
    cv.fit(training_text)

    X_train = cv.transform(training_text)

    X_train = X_train.todense()
    return X_train


# Given samples of training text, for each sample convert it into numerical
# representations (vectors) using TF-IDF
def TFIDF(training_text):
    tfidf = TfidfVectorizer(max_features = MAX_FEATURES)
    tfidf.fit(training_text)
    
    X_train = tfidf.transform(training_text)
    X_train = X_train.todense()

    return X_train


# Take an average of the word vectors to get a sentence vector
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



# Given samples of training text, for each sample convert it into numerical
# representations (vectors) using Word2Vec
def word2vec(training_text, lstm=False):
    modelTrain = gensim.models.KeyedVectors.load(
            "../fake-news/train_word2vec_model.bin")
    if lstm is True:
        paras = []
        for sample in training_text:
            # clip sample at 200 words
            sample = sample.split()
            sample = sample[:200]
            embeds = np.empty((0, 100), np.float32)
            for i in range(200):
                if len(sample) <= i:
                    embeds = np.vstack((embeds, np.zeros(100, np.float32)))
                else:
                    word = sample[i]
                    try:
                        # modelTrain[word] is a vector of shape (100, )
                        embeds = np.vstack((embeds, modelTrain[word]))
                    except KeyError:
                        embeds = np.vstack((embeds, np.zeros(100, np.float32)))

            # Now the shape of embeds is (250, 100)
            # stack it into 3-d
            paras.append(embeds)
        return np.array(paras)
    else:
        X_train = [getVector(modelTrain,sent.split(' ')) for sent in training_text]
        return np.array(X_train)

