import readdata
from text_vectorizer import word2vec, TFIDF, CV
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers, Model
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from outlier_remove import removeOutliers, getRemovedVals
from sklearn.model_selection import train_test_split
from roc import save_y
import sys
import numpy as np
from pathlib import Path


# Usage: 
# python lstm.py <model> <grid-search step / 0>
# <model> can be: cv, tfidf, or word2vec
# The last paramter can be 0 or 'grid-search step'
# 0 means actual run
# <grid-search step> can be: 1, 2, 3 to do a grid search


# For the first grid search step, do:
# 1. python lstm.py cv 1 
# 2. python lstm.py tfidf 1
# 3. python lstm.py word2vec 1


# 3 activations for hidden (btwn LSTM and final): RELU, linear, sigmoid
# 
# 2 optimizers: Adam, SGD
#
# number of hidden layers: 1, 2
# number of hidden neurons: 200, 400, 600
# number of memory cells: 200, 400, 600
# number of memory cells = length of vector 'a' in lstm layer



# Grid search steps:
# 1. search the best activations, using 1 layer, 400 neurons, 600 cells, Adam
# 2. search the best optimizer, with the best activations found in step 1,
#    and other hyper-para as step 1
# 3. search the best hidden layer and hidden neurons and the best memory cells



def create_model(look_back=None, input_nodes=None, activation='relu', 
                optimizer='adam', hidden_layers=1, neurons=400, memcells=600):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(memcells, dropout=0.2, 
                                input_shape=(look_back, input_nodes)))
    
    for _ in range(hidden_layers):
        model.add(keras.layers.Dense(neurons, activation=activation))

    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, 
                    metrics=['accuracy'])
    return model




def get_param_grid():
    grid_step = int(sys.argv[2])
    if grid_step == 1:
        activation = ['relu', 'linear', 'sigmoid']
        return dict(activation=activation)
    elif grid_step == 2:
        optimizer = ['Adam', 'SGD']
        return dict(optimizer=optimizer)
    elif grid_step == 3:
        neurons = [200, 400, 600]
        hidden_layers = [1, 2]
        memcells = [200, 400, 600]
        return dict(neurons=neurons, hidden_layers=hidden_layers,
                    memcells=memcells)
    else:
        print("Error")
        quit()



def evaluate(grid_result):
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))  


def main():
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy()



    if sys.argv[1] == "cv":
        X_train, _ = CV(X_train, X_test) # train shape: (17973, 10000)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
        # X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)

    elif sys.argv[1] == 'tfidf':
        X_train, _ = TFIDF(X_train, X_test) # train shape: (17973, 10000)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "TFIDF_Train",isTest = False)
        # X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)

    elif sys.argv[1] == 'word2vec':
        X_train, _ = word2vec(X_train, X_test)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "W2V_Train",isTest = False)
        # X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
    else:
        print("Error")
        return

    look_back = 1
    # reshape input to be [samples, time steps, features]
    num_samples = X_train.shape[0]
    num_features = X_train.shape[1]
    X_train = np.reshape(np.array(X_train), (num_samples, look_back, num_features))


    epochs = 5
    batch_size = 256


    if int(sys.argv[2]) == 0: # actual run
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, Y_train, random_state = 1, test_size = 0.34)
        model = create_model(look_back=look_back, input_nodes=num_features)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        print("----Start Evaluating----")
        _, acc = model.evaluate(X_test, y_test)
        print("Testing Accuracy:", acc)

        y_pred = model.predict(X_test)

        # Store y_pred vector
        save_y(sys.argv[1], "lstm_y_pred", y_pred)

        # Store y_true vector (Only one script needs this)
        y_true_file = Path("./model_Ys/true/y_true.npy")
        if not y_true_file.is_file():
            save_y("true", "y_true", y_test)


    else: # doing grid search
        model = KerasClassifier(build_fn=create_model, look_back=look_back, 
                    input_nodes=num_features, epochs=epochs, 
                    batch_size=batch_size, verbose=1)
        param_grid = get_param_grid()

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(X_train, Y_train)
        evaluate(grid_result)  
    



if __name__ == "__main__":
    main()