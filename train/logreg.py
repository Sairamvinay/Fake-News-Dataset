import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from outlier_remove import removeOutliers, getRemovedVals
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import sys
import numpy as np
from roc import save_y
from pathlib import Path

# Usage:
# 1. python logreg.py cv <flag>
# 2. python logreg.py tfidf <flag>
# 3. python logreg.py word2vec <flag>

# <flag>: 0 means actual run, 1 means grid search


def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH)

    X = dfTrain['text'].to_numpy()
    y = dfTrain['label'].to_numpy()

    if sys.argv[1] == "cv":
        X = CV(X)  # train shape: (17973, 10000)
        X, y = getRemovedVals(X = X,Y = y,Ftype = "CV_Train")

    elif sys.argv[1] == 'tfidf':
        X = TFIDF(X) # train shape: (17973, 10000)
        X, y = getRemovedVals(X = X,Y = y,Ftype = "TFIDF_Train")

    elif sys.argv[1] == 'word2vec':
        X = word2vec(X)
        X, y = getRemovedVals(X = X,Y = y,Ftype = "W2V_Train")
        
    else:
        print("Error")
        return

    # reshape input to be [samples, features]
    num_samples = X.shape[0]
    num_features = X.shape[1]
    X = np.reshape(np.array(X), (num_samples, num_features))

    if int(sys.argv[2]) == 0: # actual run
        C = None # to be set to the best hyperpara
        solver = None # to be set to the best hyperpara
        kf = KFold(n_splits=3, random_state=1)
        logistic = LogisticRegression(max_iter=500, C=C, solver=solver)
        acc_list = []
        # Doing cross validation testing
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            logistic.fit(X_train, y_train)
            print("----Start Evaluating----")
            acc = logistic.score(X_test, y_test)
            acc_list.append(acc)
            print("Testing Accuracy:", acc)
        print("Mean testing accuracy:", sum(acc_list) / len(acc_list))
        
        y_pred = logistic.predict(X_test)

        # Store y_pred vector
        save_y(sys.argv[1], "logreg_y_pred", y_pred)
        
        
    else: # grid search
        # creating space for constant C
        c = np.logspace(0, 4, 10)
        # various solvers - only used solvers that supported L2
        solver = ['newton-cg', 'sag', 'lbfgs', 'liblinear']
        param_grid = dict(C=c, solver=solver)
        logistic = LogisticRegression(max_iter=500)
        grid = GridSearchCV(logistic, param_grid=param_grid, cv=3, verbose=0)
        grid_result = grid.fit(X, y)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    main()
