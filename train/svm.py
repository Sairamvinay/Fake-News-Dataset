import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from text_vectorizer import outlierDection
from OutlierDetectRemove import removeOutliers
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys

# Usage: 
# 1. python lstm.py cv
# 2. python lstm.py tfidf
# 3. python lstm.py word2vec

def getRemovedVals(X,Y = None,Ftype = "",isTest = False):

    X = np.array(X)
    index,_ = outlierDection(X,Ftype)
    if not isTest:
        Y = np.array(Y)
        Xrem,Yrem = removeOutliers(index,X,Y,Ftype)
        return Xrem,Yrem

    else:
        Xrem = removeOutliers(index,X,Y,Ftype)
        return Xrem

def evaluate(pred, truth):
    print('Mean Absolute Error:', metrics.mean_absolute_error(truth, pred))
    print('Mean Squared Error:', metrics.mean_squared_error(truth, pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(truth, pred)))



def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy() # unlabelled

    if sys.argv[1] == "cv":
        X_train, X_test, _ = CV(X_train, X_test) # train shape: (17973, 141221)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)


    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 141221)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "TFIDF_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)


    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "W2V_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
        
    else:
        print("Error")
        return
    
    print("Performing Grid Search on SVM...")
    svm = SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':(0.25,0.5,0.75)}
    grid = GridSearchCV(estimator = svm, param_grid = parameters,n_jobs=-1, cv=3,verbose=1)
    grid_result = grid.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main()