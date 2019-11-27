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
textblob, gensim, re, langid,collections

## Usage

### Directories

* fake-news: contains our dataset

* preprocessing: contains scripts that we use to preprocess our data (stop
word removal, punctuation removal, etc.) and to plot the graphs that will be
useful for our analysis (sentiment analysis, Pos Tags Distribution, Unigrams and Bigrams) at the preprocessing
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

Models will be run inside the `train/` folder.
A general command will be: `python3 <fine-tuning algo> <pre-training> <flag>`


`<flag>`: can be 0 or other numbers. Other numbers mean performing the 
grid search. 0 means doing an actual run to get the testing accuracy with
a k-fold of 3 and with all the best hyperparameters set manually. 
The best hyperparameters are obtained from grid search.
The grid search results for each combination of models are available
in the directory `train/model_results`.

For example, to run logistic regression, give the following command if you
want to run it with:
1. CountVectorizer: `python3 logreg.py cv <flag>`

2. TF-IDF: `python3 logreg.py tfidf <flag>`

3. Word2Vec: `python3 logreg.py word2vec <flag>`

Again, <flag> needs to be replaced by numbers. 0 always means an actual run.

For LSTMs and ANNs, grid search is done sequentially rather than all at once
because of the limitation of computing resource. Thus, the flag for grid search
not only includes 1, but also 2, 3, and 4. Take a closer look at the comments
in those two files and you will understand how to perform the grid search.


