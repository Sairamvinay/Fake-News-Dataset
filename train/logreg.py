import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import sys
import numpy as np

# Usage:
# 1. python logreg.py cv
# 2. python logreg.py tfidf
# 3. python logreg.py word2vec

def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH, nolabel=False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH, nolabel=True)

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy()

    if sys.argv[1] == "cv":
        X_train, X_test, _ = CV(X_train, X_test)  # train shape: (17973, 141221)
    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test)  # shape: (17973, 141221)
    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
    else:
        print("Error")
        return

    # reshape input to be [samples, features]
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    X_train = np.reshape(np.array(X_train), (num_samples, num_features))

    # creating space for constant C
    c = np.logspace(0, 4, 10)
    # various solvers - only used solvers that supported L2
    solver = ['newton-cg', 'sag', 'lbfgs', 'liblinear']
    param_grid = dict(C=c, solver=solver)
    logistic = LogisticRegression(max_iter=500)
    grid = GridSearchCV(logistic, param_grid=param_grid, cv=3, verbose=0)
    grid_result = grid.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


if __name__ == "__main__":
    main()