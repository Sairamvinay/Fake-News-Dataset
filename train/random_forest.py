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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
from roc import save_y
from pathlib import Path
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
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH)

    X = dfTrain['text'].to_numpy()
    y = dfTrain['label'].to_numpy()


    if sys.argv[1] == "cv":
        X = CV(X) # train shape: (17973, 10000)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "CV_Train")


    elif sys.argv[1] == 'tfidf':
        X = TFIDF(X) # train shape: (17973, 10000)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "TFIDF_Train")


    elif sys.argv[1] == 'word2vec':
        X = word2vec(X)
        X,y = getRemovedVals(X = X,Y = y,Ftype = "W2V_Train")
        
    else:
        print("Error")
        return


    if int(sys.argv[2]) == 0: # actual run
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state = 1, test_size = 0.34)

        # These are the best hyper-para from the results of grid search
        max_depth = 9
        min_samples_leaf = 4
        min_samples_split = 5
        n_estimators = 400
        acc_list = []
        X_train = None # init
        X_test = None # init

        kf = KFold(n_splits=3, random_state=1)

        for train_index, test_index in kf.split(X):
            # Doing cross validation testing
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = RandomForestClassifier(max_depth=max_depth, 
                min_samples_leaf = min_samples_leaf, 
                min_samples_split=min_samples_split,
                n_estimators=n_estimators, n_jobs=-1)
            model.fit(X_train, y_train)
            print("----Start Evaluating----")
            acc = model.score(X_test, y_test)
            acc_list.append(acc)
            print("Testing Accuracy:", acc)
        print("Mean testing accuracy:", sum(acc_list) / len(acc_list))

        y_pred = model.predict(X_test)
        # Store y_pred vector
        save_y(sys.argv[1], "random_forest_y_pred", y_pred)
    
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
        grid_result = grid.fit(X, y)
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
