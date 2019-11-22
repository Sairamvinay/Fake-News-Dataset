import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from outlier_remove import removeOutliers, getRemovedVals
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys

# Usage: 
# 1. python svm.py cv
# 2. python svm.py tfidf
# 3. python svm.py word2vec


def evaluate(pred, truth):
    print('Mean Absolute Error:', metrics.mean_absolute_error(truth, pred))
    print('Mean Squared Error:', metrics.mean_squared_error(truth, pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(truth, pred)))



def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)

    X = dfTrain['text'].to_numpy()
    y = dfTrain['label'].to_numpy()

    if sys.argv[1] == "cv":
        X = CV(X) # train shape: (17973, 10000)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "CV_Train",isTest = False)


    elif sys.argv[1] == 'tfidf':
        X = TFIDF(X) # shape: (17973, 10000)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "TFIDF_Train",isTest = False)


    elif sys.argv[1] == 'word2vec':
        X = word2vec(X)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "W2V_Train",isTest = False)
        
    else:
        print("Error")
        return
    
    print("Performing Grid Search on SVM...")
    svm = SVC()
    parameters = {'kernel':('linear', 'rbf'), 'C':(0.25,0.5,0.75)}
    grid = GridSearchCV(estimator = svm, param_grid = parameters,n_jobs=-1, cv=3,verbose=1)
    grid_result = grid.fit(X, y)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main()