import numpy as np
from fileprocess import read_files,TRAINFILEPATH,TESTFILEPATH
from time import time
import numpy as np
from text_vectorizer import outlierDection,word2vec,TFIDF,CV
import matplotlib.pyplot as plt


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





def main():

    start = time()
    dfTrain = read_files(TRAINFILEPATH,nolabel = False)
    dfTest = read_files(TESTFILEPATH,nolabel = True)

    Y_train = dfTrain["label"]
    
    lines_length = len(dfTrain.values)
    lines_testlength = len(dfTest.values)
    
    trainVal = dfTrain["text"].values
    testVal = dfTest["text"].values
    
    training_text = [trainVal[i] for i in range(lines_length)]
    testing_text = [testVal[i] for i in range(lines_testlength)]
    
    X_train_TFIDF,X_test_TFIDF,_ = TFIDF(training_text,testing_text)
    X_train_CV,X_test_CV,_ = CV(training_text,testing_text)
    X_train_WV,X_test_WV = word2vec(training_text,testing_text)

    print(X_train_WV.shape, " is the X_train shape")
    print(X_test_WV.shape, " is the X_test shape")


    print(X_train_TFIDF.shape," is the X_train shape")
    print(X_test_TFIDF.shape," is the X_test shape")
    

    print(X_train_CV.shape," is the X_train shape")
    print(X_test_CV.shape," is the X_test shape")
    

    result1,perCVtrain = outlierDection(X_train_CV,"CV train")
    result2,perCVtest = outlierDection(X_test_CV,"CV test")
    result3,perTFIDFtrain = outlierDection(X_train_TFIDF,"TFIDF train")
    result4,perTFIDFtest = outlierDection(X_test_TFIDF,"TFIDF test")
    result5,perWVtrain = outlierDection(X_train_WV,"WV train")
    result6,perWVtest = outlierDection(X_test_WV,"WV test")

    trainOutliers = [perCVtrain,perTFIDFtrain,perWVtrain]
    testOutliers = [perCVtest,perTFIDFtest,perWVtest]
    graphOutliers(trainOutliers,testOutliers)


    end = time()
    taken = (end - start) / 60.00
    print("Time taken :%f minutes"%taken)



if __name__ == '__main__':

    main()
