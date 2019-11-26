import numpy as np
import readdata
from time import time
import numpy as np
from text_vectorizer import word2vec,TFIDF,CV
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import itertools

RNG = np.random.RandomState(42)

def getRemovedVals(X,Y = None,Ftype = "",isTest = False):
    # outlier removals of values
    X = np.array(X)
    index,_ = outlierDetection(X,Ftype)
    if not isTest:
        Y = np.array(Y)
        Xrem,Yrem = removeOutliers(index,X,Y,Ftype)
        return Xrem,Yrem

    else:
        Xrem = removeOutliers(index,X,Y,Ftype)
        return Xrem


def outlierDetection(Features,Ftype):
    # outlier detection using IsolationForest
    clf_Iso = IsolationForest(random_state=RNG,n_jobs = -1)
    clf_Iso.fit(Features)
    y_Iso_Forest = clf_Iso.predict(Features)
    result = np.where(y_Iso_Forest == -1)
    result = list(itertools.chain.from_iterable(result))
    percentOutlier = 100.00 *  np.shape(result)[0]/np.shape(y_Iso_Forest)[0]
    return result,percentOutlier


def graphOutliers(train,x = ["CV","TFIDF","W2V"]):
	num_grps = len(x)
	fig, ax = plt.subplots()
	x_pos = np.arange(num_grps)
	bar_width = 0.35
	plt.bar(x_pos, train,bar_width,alpha=0.5,color = 'r',label = "Training outlier percentage")
	plt.xlabel("Type of Encoding used")
	plt.ylabel("Outlier percentage")
	plt.title('Outlier percentage for each vectorizer')
	plt.xticks(x_pos + (bar_width/2),x)
	plt.legend()
	plt.show()

def removeOutliers(index,X,Y = None,Ftype = "CV train"):
    X_removed = np.delete(X,index,axis = 0)
    if Y is None:
        return X_removed

    else:
        Y_removed = np.delete(Y,index,axis = 0)
        return X_removed,Y_removed


def main():

    start = time()

    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    Y_train = dfTrain["label"].to_numpy()

    lines_length = len(dfTrain.values)
    
    trainVal = dfTrain["text"].values
    
    training_text = [trainVal[i] for i in range(lines_length)]
    
    X_train_TFIDF= TFIDF(training_text)
    X_train_CV = CV(training_text)
    X_train_WV = word2vec(training_text)

    trainVal = dfTrain["text"].values

    training_text = [trainVal[i] for i in range(lines_length)]

    X_train_TFIDF = TFIDF(training_text)
    X_train_CV = CV(training_text)
    X_train_WV = word2vec(training_text)

    X_train_TFIDF = np.array(X_train_TFIDF)

    X_train_CV = np.array(X_train_CV)


    X_train_WV = np.array(X_train_WV)

    print("\nFor W2V\n")
    print(X_train_WV.shape, " is before removal the X_train shape")

    print("\nFor TFIDF\n")
    print(X_train_TFIDF.shape," is before removal the X_train shape")
    
    
    print("\nFor CV\n")
    print(X_train_CV.shape," is before removal the X_train shape")


    result1,perCVtrain = outlierDetection(X_train_CV,"CV train")
    result3,perTFIDFtrain = outlierDetection(X_train_TFIDF,"TFIDF train")
    result5,perWVtrain = outlierDetection(X_train_WV,"WV train")

    trainOutliers = [perCVtrain,perTFIDFtrain,perWVtrain]
    graphOutliers(trainOutliers)
  

    end = time()
    taken = (end - start) / 60.00
    print("Time taken :%f minutes"%taken)



if __name__ == '__main__':

    #main()
    pass
