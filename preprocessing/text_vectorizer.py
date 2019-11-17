from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from sklearn.ensemble import IsolationForest
import numpy as np
import itertools


from fileprocess import read_files,TRAINFILEPATH,TESTFILEPATH
from time import time
import numpy as np


# '''
# Parameters: 2 arrays of strings which contain the text of each article within the dataset for train and test
# Returns: 3 arrays: the feature vector for train, the feature vector for test, words (the features fitted on train)
# '''
def CV(training_text,testing_text):
    cv = CountVectorizer()
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
    tfidf = TfidfVectorizer()
    tfidf.fit(training_text)
    
    X_train = tfidf.transform(training_text)
    X_test = tfidf.transform(testing_text)
    words = tfidf.get_feature_names()

    X_train = X_train.todense()
    X_test = X_test.todense()

    return X_train,X_test,words

_vectorLookup = {}
def getVector(model,tokens,size = 100):
    if model not in _vectorLookup:
        _vectorLookup[model] = {}
    vec = np.zeros(size)
    count = 0
    for word in tokens:
        try:
            if word not in _vectorLookup[model]:
                _vectorLookup[model][word] = model[word]
            vec += _vectorLookup[model][word]
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


def main():

    start = time()
    dfTrain = read_files(TRAINFILEPATH,nolabel = False)
    dfTest = read_files(TESTFILEPATH,nolabel = True)

    Y_train = dfTrain["label"]
    
    lines_length = len(dfTrain.values)
    lines_testlength = len(dfTest.values)
    
    trainVal = dfTrain["text"].values
    testVal = dfTest["text"].values
    
    training_text = [trainVal[i] for i in range(lines_length)]
    testing_text = [testVal[i] for i in range(lines_testlength)]
    
    X_train_TFIDF,X_test_TFIDF,_ = TFIDF(training_text,testing_text)
    X_train_CV,X_test_CV,_ = CV(training_text,testing_text)

    X_train_WV,X_test_WV = word2vec(training_text,testing_text)

    print(X_train_WV.shape, " is the X_train shape")
    print(X_test_WV.shape, " is the X_test shape")


    print(X_train_TFIDF.shape," is the X_train shape")
    print(X_train_TFIDF)
    print(X_test_TFIDF.shape," is the X_test shape")
    print(X_test_TFIDF)

    print(X_train_CV.shape," is the X_train shape")
    print(X_train_CV)
    print(X_test_CV.shape," is the X_test shape")
    print(X_test_CV)

    rng = np.random.RandomState(42)
    clf_Iso = IsolationForest(max_samples='auto',random_state=rng, contamination='auto', behaviour='new')
    
    clf_Iso.fit(X_train_CV)
    y_CV_Iso_Forest = clf_Iso.predict(X_train_CV)
    result_CV = np.where(y_CV_Iso_Forest == -1)
    result_CV = list(itertools.chain.from_iterable(result_CV))
    print(np.shape(result_CV))
    print(np.shape(y_CV_Iso_Forest))
    print("the percentage of outliers in CV train is", np.shape(result_CV)[0]/np.shape(y_CV_Iso_Forest)[0])

    clf_Iso.fit(X_test_CV)
    y_test_CV_Iso_Forest = clf_Iso.predict(X_test_CV)
    result_test_CV = np.where(y_test_CV_Iso_Forest == -1)
    result_test_CV = list(itertools.chain.from_iterable(result_test_CV))
    print(np.shape(result_test_CV))
    print(np.shape(y_test_CV_Iso_Forest))
    print("the percentage of outliers in CV test is", np.shape(result_test_CV)[0]/np.shape(y_test_CV_Iso_Forest)[0])
    
    clf_Iso.fit(X_train_TFIDF)
    y_TFIDF_Iso_Forest = clf_Iso.predict(X_train_TFIDF)
    result_TFIDF = np.where(y_TFIDF_Iso_Forest == -1)
    result_TFIDF = list(itertools.chain.from_iterable(result_TFIDF))
    print(np.shape(result_TFIDF))
    print(np.shape(y_TFIDF_Iso_Forest))
    print("the percentage of outliers in TFIDF train is", np.shape(result_TFIDF)[0]/np.shape(y_TFIDF_Iso_Forest)[0])

    clf_Iso.fit(X_test_TFIDF)
    y_test_TFIDF_Iso_Forest = clf_Iso.predict(X_test_TFIDF)
    result_test_TFIDF = np.where(y_test_TFIDF_Iso_Forest == -1)
    result_test_TFIDF = list(itertools.chain.from_iterable(result_test_TFIDF))
    print(np.shape(result_test_TFIDF))
    print(np.shape(y_test_TFIDF_Iso_Forest))
    print("the percentage of outliers in TFIDF test is", np.shape(result_test_TFIDF)[0]/np.shape(y_test_TFIDF_Iso_Forest)[0])

    clf_Iso.fit(X_train_WV)
    y_WV_Iso_Forest = clf_Iso.predict(X_train_WV)
    result_WV = np.where(y_WV_Iso_Forest == -1)
    result_WV = list(itertools.chain.from_iterable(result_WV))
    print(np.shape(result_WV))
    print(np.shape(y_WV_Iso_Forest))
    print("the percentage of outliers in WV train is", np.shape(result_WV)[0]/np.shape(y_WV_Iso_Forest)[0])
    
    clf_Iso.fit(X_test_WV)
    y_test_WV_Iso_Forest = clf_Iso.predict(X_test_WV)
    result_test_WV = np.where(y_test_WV_Iso_Forest == -1)
    result_test_WV = list(itertools.chain.from_iterable(result_test_WV))
    print(np.shape(result_test_WV))
    print(np.shape(y_test_WV_Iso_Forest))
    print("the percentage of outliers in WV test is", np.shape(result_test_WV)[0]/np.shape(y_test_WV_Iso_Forest)[0])

    end = time()
    taken = (end - start) / 60.00
    print("Time taken :%f minutes"%taken)



if __name__ == '__main__':

    main()







