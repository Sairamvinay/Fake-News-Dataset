import numpy as np
import readdata
from time import time
import numpy as np
from text_vectorizer import outlierDection,word2vec,TFIDF,CV
import matplotlib.pyplot as plt


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


def graphOutliers(train,test,x = ["CV","TFIDF","W2V"]):
	
	num_grps = len(x)
	fig, ax = plt.subplots()
	x_pos = np.arange(num_grps)
	bar_width = 0.35
	plt.bar(x_pos, train,bar_width,alpha=0.5,color = 'r',label = "Training outlier percentage")
	plt.bar(x_pos + bar_width, test,bar_width,alpha=0.5,color = 'b',label = "Testing outlier percentage")
	plt.xlabel("Type of Encoding used")
	plt.ylabel("Outlier percentage")
	plt.title('Outlier percentage for each vectorizer')
	plt.xticks(x_pos + (bar_width/2),x)
	plt.legend()
	plt.show()

def removeOutliers(index,X,Y = None,Ftype = "CV train"):
    X_removed = np.delete(X,index,axis = 0)
    # print(X_removed.shape," is shape of X for ", Ftype,"after removing outliers")
    if Y is None:
        return X_removed

    else:
        Y_removed = np.delete(Y,index,axis = 0)
        # print(Y_removed.shape," is shape of Y for ",Ftype," after removing outlier")
        return X_removed,Y_removed


def main():

    start = time()

    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    dfTest = readdata.read_clean_data(readdata.TESTFILEPATH,nolabel = True)
    Y_train = dfTrain["label"].to_numpy()
    
    lines_length = len(dfTrain.values)
    lines_testlength = len(dfTest.values)
    
    trainVal = dfTrain["text"].values
    testVal = dfTest["text"].values
    
    training_text = [trainVal[i] for i in range(lines_length)]
    testing_text = [testVal[i] for i in range(lines_testlength)]
    
    X_train_TFIDF,X_test_TFIDF,_ = TFIDF(training_text,testing_text)
    X_train_CV,X_test_CV,_ = CV(training_text,testing_text)
    X_train_WV,X_test_WV = word2vec(training_text,testing_text)

    X_train_TFIDF = np.array(X_train_TFIDF)
    X_test_TFIDF = np.array(X_test_TFIDF)

    X_train_CV = np.array(X_train_CV)
    X_test_CV = np.array(X_test_CV)


    X_train_WV = np.array(X_train_WV)
    X_test_WV = np.array(X_test_WV)

    print("\nFor W2V\n")
    print(X_train_WV.shape, " is before removal the X_train shape")
    print(X_test_WV.shape, " is before removal the X_test shape")

    print("\nFor TFIDF\n")
    print(X_train_TFIDF.shape," is before removal the X_train shape")
    print(X_test_TFIDF.shape," is before removal the X_test shape")
    
    
    
    print("\nFor CV\n")
    print(X_train_CV.shape," is before removal the X_train shape")
    print(X_test_CV.shape," is before removal the X_test shape")
    

    result1,perCVtrain = outlierDection(X_train_CV,"CV train")
    result2,perCVtest = outlierDection(X_test_CV,"CV test")
    result3,perTFIDFtrain = outlierDection(X_train_TFIDF,"TFIDF train")
    result4,perTFIDFtest = outlierDection(X_test_TFIDF,"TFIDF test")
    result5,perWVtrain = outlierDection(X_train_WV,"WV train")
    result6,perWVtest = outlierDection(X_test_WV,"WV test")

    trainOutliers = [perCVtrain,perTFIDFtrain,perWVtrain]
    testOutliers = [perCVtest,perTFIDFtest,perWVtest]
    graphOutliers(trainOutliers,testOutliers)
    
    
    X_train_CV,Y_train_CV = removeOutliers(index = result1,X = X_train_CV,Y = Y_train,Ftype = "CV train")
    X_test_CV = removeOutliers(index = result2,X = X_test_CV,Y = None,Ftype = "CV test")


    X_train_TFIDF,Y_train_TFIDF = removeOutliers(index = result3,X = X_train_TFIDF,Y = Y_train,Ftype = "TFIDF train")
    X_test_TFIDF = removeOutliers(index = result4,X = X_test_TFIDF,Y = None,Ftype = "TFIDF test")

    X_train_WV,Y_train_WV = removeOutliers(index = result5,X = X_train_WV,Y = Y_train,Ftype = "W2V train")
    X_test_WV = removeOutliers(index = result6,X = X_test_WV,Y = None,Ftype = "W2V test")


    

    end = time()
    taken = (end - start) / 60.00
    print("Time taken :%f minutes"%taken)



if __name__ == '__main__':

    #main()
    pass
