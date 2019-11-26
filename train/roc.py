from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import sys

# Usage: python roc.py <model-name>
# <model-name>: cv, tfidf, word2vec

# model_name is either cv, tfidf, or word2vec
def graph_roc(model_name):	
	y_scores, algo_names = load_all_y_pred(model_name)
	y_true = load_y('true', 'y_true_' + model_name)
	colors = ['red','blue','green','cyan','orange']
	i = 0
	for score, algo_name in zip(y_scores, algo_names):
		fpr, tpr, _ = roc_curve(y_true, score)
		roc_auc = auc(fpr, tpr)
		label = "ROC curve for %s " % (algo_name)
		plt.plot(fpr, tpr, color=colors[i],
				lw=2, label=label)

		i += 1

	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example for %s'%model_name.upper())
	plt.legend(loc="lower right")
	plt.show()


# Save the predicted vector of y
# Every algorithm(lstm, ann, etc.) needs to store their own y_pred
# into an unique file. y_true will only be stored once, since there
# exists only 1 y_true. So most of you will pass in filename and y_pred.
# The filename needs to be special for each algo.
# Filename should be one of the following:
# ann_y_pred
# lstm_y_pred
# svm_y_pred
# random_forest_y_pred
# logreg_y_pred
# y_true (Don't worry about this one)
#
# model_name should be one of the following:
# tfidf, cv, word2vec
def save_y(model_name, filename, y_pred, y_true=None):
	path = './model_Ys/' + model_name + '/' + filename
	np.save(path, y_pred)
	if y_true is not None:
		np.save(path, y_true)


def load_y(model_name, filename):
	path = './model_Ys/' + model_name + '/' + filename + '.npy'
	return np.load(path)


def load_all_y_pred(model_name):
	y_list = []
	filenames = ['ann_y_pred', 'lstm_y_pred', 'svm_y_pred', 
					'random_forest_y_pred', 'logreg_y_pred']
	for f in filenames:
		y_list.append(load_y(model_name, f))
	
	model_names = ['ANN', 'LSTM', 'SVM', 'Random Forest', 'Log Reg']
	return y_list, model_names


def main():
	if sys.argv[1] == "cv":
		graph_roc("cv")
	elif sys.argv[1] == "tfidf":
		graph_roc("tfidf")
	elif sys.argv[1] == "word2vec":
		graph_roc("word2vec")
	else:
		print("Error")
		return

if __name__ == "__main__":
	main()