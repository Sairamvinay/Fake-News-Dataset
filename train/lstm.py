import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
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


# for grid search
def grid_model(look_back=None, input_nodes=None, activation='relu', optimizer='adam', hidden_layers=0, neurons=3):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(1000, dropout=0.2, input_shape=(look_back, input_nodes)))
    
    for i in range(hidden_layers):
        model.add(keras.layers.Dense(neurons, activation=activation))

    model.add(keras.layers.Dense(1, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# for model after doing the grid search
def create_model(look_back, input_nodes):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(1000, dropout=0.2, input_shape=(look_back, input_nodes)))
    model.add(keras.layers.Dense(1, activation='softmax'))
    opt = optimizers.Adam(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def getRemovedVals(X,Y = None,Ftype = "",isTest = False):

    X = np.array(X)
    index,_ = outlierDection(X,Ftype)
    if not isTest:
        Y = np.array(Y)
        Xrem,Yrem = removeOutliers(index,X,Y,Ftype)
        return Xrem,Yrem

    else:
        Xrem = removeOutliers(index,X,Y,Ftype)
        return Xrem


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
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)
        look_back = 1

    elif sys.argv[1] == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 141221)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "TFIDF_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)
        look_back = 1

    elif sys.argv[1] == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "W2V_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
        look_back = 250
        # X_train = pad_sequences(X_train, maxlen=MAX_LENGTH)
        # X_test = pad_sequences(X_test, maxlen=MAX_LENGTH)
    else:
        print("Error")
        return

    # reshape input to be [samples, time steps, features]
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    X_train = np.reshape(np.array(X_train), (num_samples, look_back, num_features))

    epochs = 2
    batch_size = 128

    # model = create_model(look_back, num_features)
    # model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

    model = KerasClassifier(build_fn=grid_model, look_back=look_back, 
                input_nodes=num_features, epochs=epochs, 
                batch_size=batch_size, verbose=0)
    activation = ['relu', 'linear', 'sigmoid']
    optimizer = ['Adam', 'SGD']
    hidden_layers = [1, 2, 3]
    neurons = [3, 6, 12]
    param_grid = dict(activation=activation, optimizer=optimizer, 
                    hidden_layers=hidden_layers, neurons=neurons)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X_train, Y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))    
    



if __name__ == "__main__":
    main()