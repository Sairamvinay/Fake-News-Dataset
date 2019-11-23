import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from outlier_remove import removeOutliers, getRemovedVals
from sklearn import metrics
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from roc import save_y
from pathlib import Path
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
    
    if int(sys.argv[2]) == 0: # actual run
        
        kf = KFold(n_splits=3, random_state=1)
        svm = SVC(max_iter=500, C=0.25)
        acc_list = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            svm.fit(X_train, y_train)
            print("----Start Evaluating----")
            acc = svm.score(X_test, y_test)
            acc_list.append(acc)
            print("Testing Accuracy:", acc)
        print("Mean testing accuracy:", sum(acc_list) / len(acc_list))
        
        y_pred = svm.predict(X_test)

        # Store y_pred vector
        save_y(sys.argv[1], "svm_y_pred", y_pred)

    
    else: # grid search
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