from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

from fileprocess import read_files,TRAINFILEPATH,TESTFILEPATH
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



def word2vec(training_text, testing_text):
    modelTrain = gensim.models.KeyedVectors.load_word2vec_format(
            "../fake-news/train_word2vec_model.bin", binary=True)

    modelTest = gensim.models.KeyedVectors.load_word2vec_format("../fake-news/test_word2vec_model.bin",binary = True)
    X_train = [modelTrain[word] for word in training_text]
    X_test = [modelTest[word] for word in testing_text]

    return X_train, X_test


def main():
    dfTrain = read_files(TRAINFILEPATH,nolabel = False)
    dfTest = read_files(TESTFILEPATH,nolabel = True)
    lines_length = len(dfTrain.values)
    lines_testlength = len(dfTest.values)
    training_text = [dfTrain["text"].values[i] for i in range(lines_length)]
    testing_text = [dfTest["text"].values[i] for i in range(lines_testlength)]
    X_train_TFIDF,X_test_TFIDF,_ = TFIDF(training_text,testing_text)
    X_train_CV,X_test_CV,_ = CV(training_text,testing_text)
   
    print(X_train_TFIDF.shape," is the X_train shape")
    print(X_test_TFIDF.shape," is the X_test shape")

    print(X_train_CV.shape," is the X_train shape")
    print(X_test_CV.shape," is the X_test shape")



if __name__ == '__main__':

    main()







