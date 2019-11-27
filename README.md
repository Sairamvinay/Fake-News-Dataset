# ECS 171 Final Project - Fake News Detection

Many news sources contain false information and are therefore “fake news.” 
Because there is a lot of “fake news” articles and fabricated,
misleading information on the web, we would like to determine which texts are
legitimate (real) and which are illegitimate (fake). To solve this as a binary 
classification problem, we investigate the effectiveness of different Natural 
Language Processing models which are used to convert character based texts into 
numeric representations such as TFIDF, CountVectorizer and Word2Vec models and 
find out which model is able to preserve most of the contextual information 
about the text used in a fake news data set and how helpful and effective it is 
in detecting whether the text is a fake news or not.


## Dataset
https://www.kaggle.com/c/fake-news/data

## Dependencies

* Languages: python3

* Libraries: tensorflow, keras, sklearn, numpy, pandas, matplotlib, spacy,
textblob, gensim, re

## Usage

### Directories

* fake-news: contains our dataset

* preprocessing: contains scripts that we use to preprocess our data (stop
word removal, puntuation removel, etc.) and to plot the graphs that will be
useful for our analysis (sentiment analysis, Pos Tags) at the preprocessing
stage.

* train: all the pre-training and fine-tuning models are here.

### Model architecture

Our model consists of two stages. In the first stages, three 
pre-training algorithms are applied to the cleaned text to convert them into
numerical representations. And at the second stage, the numerical 
representations of text are fed into five fine-tuning algorithms.

### Pre-training algorithms

* CountVectorizer

* TF-IDF

* Word2Vec

### Fine-tuning algorithms

* ANNs

* LSTMs

* Logistic Regression

* Support Vector Machine

* Random Forest Classifier

### How to run our model

A general 



Instructions to run model code: <br/>
model can be: cv, tfidf, or word2vec
	
python ann.py <model> <grid-search step / 0>,  <grid-search step> can be: 1, 2, 3, 4 to do a grid search, 0 means actual run <br/>
python logreg.py <model> <flag>,  <flag>: 0 means actual run, 1 means grid search <br/>
python lstm.py <model> <grid-search step / 0>, <grid-search step> can be: 1, 2, 3 to do a grid search, 0 means actual run <br/>
python pca.py <model-name> <flag>, <flag>: 1 to standardize, 0 to not <br/>
python random_forest.py <model> <flag>. <flag>: 0 means actual run, 1 means grid search <br/>
python roc.py <model-name> <br/>
python svm.py word2vec <flag>, flag is for the running: 0 for simple K fold and getting graph, and 1 for grid search <br/>

