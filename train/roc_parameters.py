from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

def graph_roc(y_train):
	# y_score = grid_result.predict(X_train)

	# fpr, tpr, _ = roc_curve(y_train, y_score)
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


# Save the predicted vector of y
# Every algorithm(lstm, ann, etc.) needs to store their own y_pred
# into an unique file. y_true will only be stored once, since there
# exists only 1 y_true. So most of you will pass in filename and y_pred.
# The filename needs to be special for each algo.
# Filename should be as follows:
# ann_y_pred
# lstm_y_pred
# svm_y_pred
# random_forest_y_pred
# logreg_y_pred
# y_true (Don't worry about this one)
# Note: Just pass in the filename! I will add the path for you in the code below
def save_y(filename, y_pred, y_true=None):
	path = './model_Ys/' + filename
	np.save(path, y_pred)
	if y_true is not None:
		np.save(path, y_true)


def load_y(filename):
	path = './model_Ys/' + filename + '.npy'
	return np.load(path)