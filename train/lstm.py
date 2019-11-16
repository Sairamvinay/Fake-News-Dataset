import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow import keras
import sys

# Usage: 
# 1. python lstm.py cv
# 2. python lstm.py tfidf
# 3. python lstm.py word2vec


def create_model(input_nodes):
    # model = keras.Sequential()
    # model.add(keras.layers.LSTM(1000, dropout=0.2, input_shape=(input_nodes,)))
    # model.add(keras.layers.Dense(1, activation='softmax'))
    input_layer = keras.layers.Input(shape=(input_nodes, ))
    hidden1 = keras.layers.LSTM(1000, dropout=0.2)(input_layer)
    final = keras.layers.Dense(1, activation='softmax')(hidden1)
    model = Model(input_layer, final)
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
    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 141221)
    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
    else:
        print("Error")
        return

    # pad_sequences(X_train, maxlen=MAX_LENGTH)
    # pad_sequences(X_test, maxlen=MAX_LENGTH)
    epochs = 5
    batch_size = 128
    model = create_model(X_train.shape[1])
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)
    



if __name__ == "__main__":
    main()