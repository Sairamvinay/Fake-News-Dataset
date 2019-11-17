import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow import keras
import sys
import numpy as np

# Usage: 
# 1. python lstm.py cv
# 2. python lstm.py tfidf
# 3. python lstm.py word2vec


# 3 activations for hidden (btwn LSTM and final): RELU, linear, sigmoid
# 2 optimizers: Adam, SGD
#
#
# number of hidden layers: 0(default), 1, 2, 3
# number of hidden neurons: 3, 6, 12

# separate
# number of memory cells: 200, 400, 600

def create_model(look_back, input_nodes):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(1000, dropout=0.2, input_shape=(look_back, input_nodes)))
    model.add(keras.layers.Dense(1, activation='softmax'))
    opt = optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def main():
    # Max number of words in each X vector
    MAX_LENGTH = 250
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy()



    if sys.argv[1] == "cv":
        X_train, X_test, _ = CV(X_train, X_test) # train shape: (17973, 141221)
        look_back = 1
    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 141221)
        look_back = 1
    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        look_back = 250
        X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
        X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)
    else:
        print("Error")
        return

    # reshape input to be [samples, time steps, features]
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    X_train = np.reshape(np.array(X_train), (num_samples, look_back, num_features))
    # pad_sequences(X_train, maxlen=MAX_LENGTH)
    # pad_sequences(X_test, maxlen=MAX_LENGTH)
    epochs = 5
    batch_size = 128
    model = create_model(look_back, num_features)
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    



if __name__ == "__main__":
    main()