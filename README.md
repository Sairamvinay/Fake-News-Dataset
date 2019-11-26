# Fake-News-Dataset
ECS 171 Final Project

REQUIREMENTS:


DataSet:  https://www.kaggle.com/c/fake-news/data

NOTE: 

train.csv: A full training dataset with the following attributes:

id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
1: unreliable
0: reliable
test.csv: A testing training dataset with all the same attributes at train.csv without the label.

submit.csv: A sample submission that you can


We need to write a report on the project by November 21, 2019. The report should contain these sections: General Information, Abstract, Introduction, Method (Used by Our team), Results, Discussion, Conclusion, References.


Hypothesis: We will investigate the effectiveness of different Natural Language Processing Models such as TFIDF, CountVectorizer and Word2Vec models and find out which model is able to preserve most of the information about the text used in fake news data set and how helpful is it in detecting whether is it a fake news or not.


Method: 3 Phases in particular to be executed:

Phase 1: Data Preprocessing and Model Creation
Estimated Completion date: Nov. 9

Data Cleaning: Need to extract text from each sample, need to remove stopwords, punctuation, numbers and so on. We need to clean the data before we begin.
		
		https://medium.com/@makcedward/nlp-pipeline-stop-words-part-5-d6770df8a936
		

Need to then create 3 different models using all the texts: TFIDF, CountVectorizer and W2V. 
				https://www.freecodecamp.org/news/how-to-process-textual-data-using-tf-idf-in-python-cd2bbc0a94a3/

https://medium.com/greyatom/an-introduction-to-bag-of-words-in-nlp-ac967d43b428

https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/



Then we perform Outlier Detection and remove missing value records and outlier records.

Finally we normalize our data set.



Phase 2: Fake News Detector Evaluation
Estimated Completion date: Nov. 16

We plan to use 5 algorithms (subject to change): ANN, RNN (preferably LSTM), SVM, Logistic Regression, and Random Forest. 

NOTE: All these evaluations are done ONLY on training data set for now. Since Test data set has no labels.

We run these 5 algorithms on each of the 3 models (15 in total): 

We first do a Grid Search on all the hyperparameters for each of the models (15 of them) and we will do K fold Cross Validation on each of the models with the best hyperparameters.

We will plot graphs for each of them, obtain the required results and present tables and graphs. We need to show precision, recall, accuracy and F1 scores.

We need to look into ROC, PR curves as mentioned in the prompt.


Phase 3: Concluding and Evaluation on the Test (Unlabelled) data set.
Estimated Completion date: Nov. 17

We choose the best model and evaluate the model on the test by training on the whole training data set.
We present that model as our best fake news detector model and we showcase our results.

Phase 4: Related Work
Should do this side by side

Need to find state of the art papers on the SAME TOPICpython roc.py <model-name>
python svm.py word2vec <flag>, # flag is for the running: 0 for simple K fold and getting graph, and 1 for grid searchand we need to show what’s been done before and what we have done.

Phase 5: Report Writing
Due date: Nov. 21

Needs to be in a research paper format. Good formatting with graphs and results.

Instructions to run model code: <br/>
model can be: cv, tfidf, or word2vec
	
python ann.py <model> <grid-search step / 0>,  <grid-search step> can be: 1, 2, 3, 4 to do a grid search, 0 means actual run <br/>
python logreg.py <model> <flag>,  <flag>: 0 means actual run, 1 means grid search <br/>
python lstm.py <model> <grid-search step / 0>, <grid-search step> can be: 1, 2, 3 to do a grid search, 0 means actual run <br/>
python pca.py <model-name> <flag>, <flag>: 1 to standardize, 0 to not <br/>
python random_forest.py <model> <flag>. <flag>: 0 means actual run, 1 means grid search <br/>
python roc.py <model-name> <br/>
python svm.py word2vec <flag>, flag is for the running: 0 for simple K fold and getting graph, and 1 for grid search <br/>

