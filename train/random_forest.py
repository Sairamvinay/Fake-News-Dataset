import readdata
from sklearn.ensemble import RandomForestClassifier
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import KFold
import numpy as np
import sys

# Usage: 
# 1. python lstm.py cv
# 2. python lstm.py tfidf
# 3. python lstm.py word2vec



def evaluate(pred, truth):
    print('Mean Absolute Error:', metrics.mean_absolute_error(truth, pred))
    print('Mean Squared Error:', metrics.mean_squared_error(truth, pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(truth, pred)))



def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)

    X = dfTrain['text'].to_numpy()
    Y = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy() # unlabelled


    if sys.argv[1] == "cv":
        X, X_test, _ = CV(X, X_test) # train shape: (17973, 141221)
    elif sys.argv[1] == 'tfidf':
        X, X_test, _ = TFIDF(X, X_test) # shape: (17973, 141221)
    elif sys.argv[1] == 'word2vec':
        X, X_test = word2vec(X, X_test)
        MAX_LENGTH = 250
        X = pad_sequences(X, maxlen=MAX_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)
    else:
        print("Error")
        return

    kf = KFold(n_splits=5)
    for train_index, dev_index in kf.split(X_train):
        X_train, X_dev = X[train_index], X[dev_index]
        y_train, y_dev = Y[train_index], Y[dev_index]   
        model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_dev)
        evaluate(y_pred, y_dev)



if __name__ == "__main__":
    main()