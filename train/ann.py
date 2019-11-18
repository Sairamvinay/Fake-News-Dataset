import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from text_vectorizer import outlierDection
from OutlierDetectRemove import removeOutliers
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import sys
import numpy as np

# Usage: 
# 1. python ann.py cv
# 2. python ann.py tfidf
# 3. python ann.py word2vec



def ANN(input_dim = 10000,num_neurons = 500,activation = "relu",hidden_layers = 3,loss = "binary_crossentropy",optimizer = "Adam",batch_size = 500,epochs = 100):
	
	
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(num_neurons,input_dim = input_dim,activation = activation))	
	
	for i in range(hidden_layers):

		model.add(keras.layers.Dense(num_neurons,activation = activation))


	model.add(keras.layers.Dense(1,activation = "softmax"))	#only binary classification
	print("Let's now compile the model")
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
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

	dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
	dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)


	X_train = dfTrain['text'].to_numpy()
	Y_train = dfTrain['label'].to_numpy()
	X_test = dfTest['text'].to_numpy()

	if sys.argv[1] == "cv":
	    X_train, X_test, _ = CV(X_train, X_test) # train shape: (17973, 10000)
	  	X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
	  	X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)
	    
	elif sys.argv[1] == 'tfidf':
	    X_train, X_test, _ = TFIDF(X_train, X_test) # shape: (17973, 10000)
	    X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "TFIDF_Train",isTest = False)
	    X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)
	    
	elif sys.argv[1] == 'word2vec':
	    X_train, X_test = word2vec(X_train, X_test)
	    X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "W2V_Train",isTest = False)
	    X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
	    
	else:
	    print("Error")
	    return

	num_samples = X_train.shape[0]
	num_features = X_train.shape[1]

	epochs = 100
	batch_size = 128

	model = KerasClassifier(build_fn=ANN,
	            input_dim = num_features, epochs = epochs, 
	            batch_size = batch_size, verbose=1)
	activation = ['relu', 'linear', 'sigmoid']
	optimizer = ['Adam', 'SGD']
	hidden_layers = [1, 2, 3]
	neurons = [3, 6, 12]

	param_grid = dict(activation=activation, optimizer=optimizer, 
	                hidden_layers=hidden_layers, num_neurons=neurons)

	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)

	grid_result = grid.fit(X_train, Y_train)
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
	    print("%f (%f) with: %r" % (mean, stdev, param))
	


if __name__ == '__main__':
	main()

