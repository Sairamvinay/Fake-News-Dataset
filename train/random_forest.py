import readdata
from sklearn.ensemble import RandomForestClassifier
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from sklearn import metrics
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy()


    if sys.argv[1] == "cv":
        X_train, X_test, _ = CV(X_train, X_test) # train shape: (17973, 141221)
    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 141221)
    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        MAX_LENGTH = 250
        X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)
    else:
        print("Error")
        return

    model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=-1)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_train)
    evaluate(y_pred, Y_train)



if __name__ == "__main__":
    main()