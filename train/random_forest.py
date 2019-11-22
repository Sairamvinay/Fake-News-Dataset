import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from outlier_remove import removeOutliers, getRemovedVals
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np
import sys

# Usage: 
# 1. python random_forest.py cv <flag>
# 2. python random_forest.py tfidf <flag>
# 3. python random_forest.py word2vec <flag>

# <flag>: 0 means actual run, 1 means grid search

# cross-validation 3
# number of estimators: 200, 400, 800
# max_depth: 1, 5, 9
# min_samples_leaf: 2, 4
# min_samples_split: 5, 10


def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)

    X_train = dfTrain['text'].to_numpy()
    y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy() # unlabelled


    if sys.argv[1] == "cv":
        X_train, _ = CV(X_train, X_test) # train shape: (17973, 10000)
        # X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
        # X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)


    elif sys.argv[1] == 'tfidf':
        X_train, X_test = TFIDF(X_train, X_test) # train shape: (17973, 10000)
        X_train,y_train = getRemovedVals(X = X_train,Y = y_train,Ftype = "TFIDF_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)


    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        X_train,y_train = getRemovedVals(X = X_train,Y = y_train,Ftype = "W2V_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
        
    else:
        print("Error")
        return


    if int(sys.argv[2]) == 0: # actual run
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, random_state = 1, test_size = 0.34)

        # These are the best hyper-para from the results of grid search
        max_depth = 40
        min_samples_leaf = 2
        min_samples_split = 5
        n_estimators = 200

        model = RandomForestClassifier(max_depth=max_depth, 
            min_samples_leaf = min_samples_leaf, 
            min_samples_split=min_samples_split,
            n_estimators=n_estimators, n_jobs=-1)
        model.fit(X_train, y_train)
        print("----Start Evaluating----")
        acc = model.score(X_train, y_train)
        print("Training Accuracy:", acc)
    
        acc = model.score(X_test, y_test)
        print("Testing Accuracy:", acc)
    
    elif int(sys.argv[2]) == 1: # grid search
        # below are the hyperparameters to be grid-searched on
        n_estimators = [200, 400, 800]
        max_depth = [1, 5, 9]
        min_samples_leaf = [2, 4]
        min_samples_split = [5, 10]
        param_grid = dict(n_estimators=n_estimators, max_depth=max_depth,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=min_samples_split)

        model = RandomForestClassifier()
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
        grid_result = grid.fit(X_train, y_train)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))
    else:
        print("Error")
        return




if __name__ == "__main__":
    main()
