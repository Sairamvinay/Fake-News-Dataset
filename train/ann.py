import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from outlier_remove import removeOutliers, getRemovedVals
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import sys
import numpy as np

# Usage: 
# python ann.py <model> <grid-search step / 0>
# <model> can be: cv, tfidf, or word2vec
# The last paramter can be 0 or 'grid-search step'
# 0 means actual run (not grid search)
# <grid-search step> can be: 1, 2, 3, 4 to do a grid search


TEST_RATIO = 0.34
EPOCHS = 25
BATCH_SIZE = 128

def get_param_grid():
    grid_step = int(sys.argv[2])
    if grid_step == 1:
        activation = ['relu', 'linear', 'sigmoid']
        return dict(activation=activation)
    elif grid_step == 2:
        optimizer = ['Adam', 'SGD']
        return dict(optimizer=optimizer)
    elif grid_step == 3:
        neurons = [200,400,600]
        hidden_layers = [1, 2, 3]
        return dict(neurons=neurons, hidden_layers=hidden_layers)
    else:
    	print("Error")
    	quit()


def ANN(input_dim = 10000,num_neurons = 500,activation = "relu",hidden_layers = 3,loss = "binary_crossentropy",optimizer = "Adam",batch_size = BATCH_SIZE,epochs = EPOCHS):
	
	
	model = keras.models.Sequential()
	model.add(keras.layers.Dense(num_neurons,input_dim = input_dim,activation = activation))	
	
	for i in range(hidden_layers):

		model.add(keras.layers.Dense(num_neurons,activation = activation))


	model.add(keras.layers.Dense(1,activation = "sigmoid"))	#only binary classification
	print("Let's now compile the model")
	model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
	return model



def main():

	dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)

	X = dfTrain['text'].to_numpy()
	y = dfTrain['label'].to_numpy()

	if sys.argv[1] == "cv":
	    X = CV(X) # train shape: (17973, 10000)
	    X,y = getRemovedVals(X = X,Y = y,Ftype = "CV_Train",isTest = False)
	    
	elif sys.argv[1] == 'tfidf':
		X = TFIDF(X) # train shape: (17973, 10000)
		X,y = getRemovedVals(X = X,Y = y,Ftype = "TFIDF_Train",isTest = False)
		
	    
	elif sys.argv[1] == 'word2vec':
		X = word2vec(X)
		X,y = getRemovedVals(X = X,Y = y,Ftype = "W2V_Train",isTest = False)
		
	    
	else:
	    print("Error")
	    return

	
	num_samples = X.shape[0]
	num_features = X.shape[1]



	if int(sys.argv[2]) == 0:
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 1, test_size = TEST_RATIO)
		model = ANN() #need to populate this with best hyperparameters after all Grid search
		model.fit(X_train,y_train,epochs = EPOCHS,batch_size = BATCH_SIZE)
		print("----Start Evaluating----")
		_, acc = model.evaluate(X_test, y_test)
		print("Testing Accuracy:", acc * 100.00,"%")
		y_pred = model.predict(X_test)
		# Store y_pred vector
		save_y(sys.argv[1], "ann_y_pred", y_pred)


	else:

		model = KerasClassifier(build_fn=ANN,
		            input_dim = num_features, epochs = EPOCHS, 
		            batch_size = BATCH_SIZE, verbose=1,activation = "sigmoid")
		

		param_grid = get_param_grid()
		grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)

		grid_result = grid.fit(X, y)
		print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
		means = grid_result.cv_results_['mean_test_score']
		stds = grid_result.cv_results_['std_test_score']
		params = grid_result.cv_results_['params']
		for mean, stdev, param in zip(means, stds, params):
		    print("%f (%f) with: %r" % (mean, stdev, param))



if __name__ == '__main__':
	main()

