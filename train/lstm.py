import readdata
from text_vectorizer import word2vec, TFIDF, CV
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import optimizers, Model
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from outlier_remove import removeOutliers, getRemovedVals
from sklearn.model_selection import KFold
from roc import save_y
import sys
import numpy as np
from pathlib import Path
from graphs_neuron_network import graphs_nn

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
                optimizer='adam', hidden_layers=0, neurons=400, hidden_units=600):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(hidden_units, dropout=0.2, 
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
        # neurons = [200, 400, 600]
        # hidden_layers = [1, 2]
        hidden_units = [200, 400, 600]
        return dict(hidden_units=hidden_units)
        # return dict(neurons=neurons, hidden_layers=hidden_layers,
        #             hidden_units=hidden_units)
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
    X = dfTrain['text'].to_numpy()
    y= dfTrain['label'].to_numpy()

    if sys.argv[1] == "cv":
        X = CV(X) # train shape: (17973, 10000)
        X, y = getRemovedVals(X = X,Y = y,Ftype = "CV_Train",isTest = False)
        look_back = 1

    elif sys.argv[1] == 'tfidf':
        X = TFIDF(X) # train shape: (17973, 10000)
        X, y = getRemovedVals(X = X,Y = y,Ftype = "TFIDF_Train",isTest = False)
        look_back = 1

    elif sys.argv[1] == 'word2vec':
        X = word2vec(X, lstm=True) # train shape: (17193, 100)
        X, y = getRemovedVals(X = X, Y = y, Ftype = "W2V_Train",isTest = False)
        look_back = X.shape[1]
        # look_back = 1

    else:
        print("Error")
        return

    num_samples = X.shape[0]

    if look_back == 1:
        # reshape input to be [samples, time steps, features]
        num_features = X.shape[1]
        X = np.reshape(np.array(X), (num_samples, look_back, num_features))
    else:
        num_features = X.shape[2]

    batch_size = 256

    if int(sys.argv[2]) == 0: # actual run
        epochs = 20 # can change this
        kf = KFold(n_splits=3, random_state=1)
        acc_list = []
        X_train = None # init
        X_test = None # init

        # X_train, X_test, y_train, y_test = train_test_split(
        #     X, y, test_size=0.33, random_state=42)
        # model = create_model(look_back=look_back, input_nodes=num_features)
        # history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
        #                         epochs=epochs, batch_size=batch_size)
        # _, acc = model.evaluate(X_test, y_test, verbose=0)
        # print("Testing Accuracy:", acc)

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model = create_model(look_back=look_back, input_nodes=num_features)
            history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                epochs=epochs, batch_size=batch_size)
            print("----Start Evaluating----")
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            acc_list.append(acc)
            print("Testing Accuracy:", acc)
        print("Mean testing accuracy:", sum(acc_list) / len(acc_list))


        loss = history.history['loss']
        val_loss = history.history['val_loss']
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']
        graphs_nn(loss, val_loss, accuracy, val_accuracy)

        y_pred = model.predict(X_test)

        # Store y_pred vector
        save_y(sys.argv[1], "lstm_y_pred", y_pred)

        # # Store y_true vector (Only one script needs this)
        # y_true_file = Path("./model_Ys/true/y_true.npy")
        # if not y_true_file.is_file():
        #     save_y("true", "y_true_" + sys.argv[1], y_test)


    else: # doing grid search
        epochs = 20
        model = KerasClassifier(build_fn=create_model, look_back=look_back, 
                    input_nodes=num_features, epochs=epochs, 
                    batch_size=batch_size, verbose=1)
        param_grid = get_param_grid()

        grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
        grid_result = grid.fit(X, y)
        evaluate(grid_result)  
    



if __name__ == "__main__":
    main()
