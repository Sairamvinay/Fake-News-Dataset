# Fake-News-Dataset
ECS 171 Final Project

REQUIREMENTS:


DataSet:  https://www.kaggle.com/c/fake-news/data


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

Need to find state of the art papers on the SAME TOPIC and we need to show what’s been done before and what we have done.

Phase 5: Report Writing
Due date: Nov. 21

Needs to be in a research paper format. Good formatting with graphs and results.
