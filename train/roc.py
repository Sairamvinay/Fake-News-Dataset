from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import readdata
from text_vectorizer import CV
from text_vectorizer import TFIDF
from text_vectorizer import word2vec
from text_vectorizer import outlierDection
from OutlierDetectRemove import removeOutliers
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
from tensorflow import keras
from tensorflow.keras import optimizers, Model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def graph_roc(grid_result):
	y_score = grid_result.predict(X_train)

	fpr, tpr, _ = roc_curve(Y_train, y_score)
	roc_auc = auc(fpr, tpr)

	lw = 2
	plt.plot(fpr, tpr, color='darkorange',
			 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()
	

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


def roc(classifier_name,ftype, model_parameters ):
    
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH, nolabel=False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH, nolabel=True)

    X_train = dfTrain['text'].to_numpy()
    Y_train = dfTrain['label'].to_numpy()
    X_test = dfTest['text'].to_numpy()
    
    if ftype == "cv":
        X_train, X_test, _ = CV(X_train, X_test)  # train shape: (17973, 141221)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "CV_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "CV_Test",isTest = True)


    elif ftype == 'tfidf':
        X_train, X_test, _ = TFIDF(X_train, X_test)  # shape: (17973, 141221)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "TFIDF_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "TFIDF_Test",isTest = True)
        
    elif ftype == 'word2vec':
        X_train, X_test = word2vec(X_train, X_test)
        X_train,Y_train = getRemovedVals(X = X_train,Y = Y_train,Ftype = "W2V_Train",isTest = False)
        X_test = getRemovedVals(X = X_test,Y = None,Ftype = "W2V_Test",isTest = True)
        
    else:
        print("Error")
        return
    
    if classifier_name == "svc":
        classifier = SVC
    elif classifier_name == "rf":
        classifier = RandomForestClassifier
    elif classifier_name == "logreg":
        classifier = LogisticRegression
    elif classifier_name == "ann":
        classifier = KerasClassifier
    elif classifier_name == "lstm":
        classifier = keras.Sequential()

    model = classifier(**model_parameters)
    model.fit(X_train, Y_train)
    y_score = model.predict(X_train)

    fpr, tpr, _ = roc_curve(Y_train, y_score)
    roc_auc = auc(fpr, tpr)

    lw = 2

    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(classifier_name + " " + ftype)
    plt.legend(loc="lower right")
    plt.show()
    
# roc( "svc", "word2vec", {'C': 0.25, 'kernel': 'linear'})
# roc("logreg", "word2vec", {'C': 10000.0, 'solver': 'newton-cg'})
# roc("rf", "tfidf",{'max_depth': 9, 'min_samples_leaf': 4, 'min_samples_split': 5, 'n_estimators': 400})