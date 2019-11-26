import readdata
from sklearn.decomposition import PCA
import pandas as pd
from text_vectorizer import word2vec, TFIDF, CV
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys

# Usage: python pca.py <model-name> <flag>
# <model-name>: tfidf, cv, word2vec
# <flag>: 1 to standardize, 0 to not

# Citation:
# https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60

def plot(finaldf, model_name):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA for ' + model_name, fontsize = 20)
    labels = [0, 1]
    colors = ['r', 'g']
    for label, color in zip(labels, colors):
        indicesToKeep = finaldf['label'] == label
        ax.scatter(finaldf.loc[indicesToKeep, 'a'],
            finaldf.loc[indicesToKeep, 'b'], c=color,s=50)
    ax.legend(labels)
    ax.grid()
    plt.show()


def main():
    # PCA feature reduction on cv, tfidf, word2vec
    dfTrain = readdata.read_clean_data(readdata.TRAINFILEPATH,nolabel = False)
    X = dfTrain['text'].to_numpy()
    if sys.argv[1] == 'cv':
        model_name = 'Count Vectorizer'
        X = CV(X)
    elif sys.argv[1] == 'tfidf':
        model_name = 'TFIDF'
        X = TFIDF(X)
    elif sys.argv[1] == 'word2vec':
        model_name = 'word2vec'
        X = word2vec(X)
    else:
        print("Error")
        return

    pca = PCA(n_components=2)
    if int(sys.argv[2]) == 1:
         X = StandardScaler().fit_transform(X)
    comp = pca.fit_transform(X)
    xdf = pd.DataFrame(data=comp, columns=['a','b'])
    finaldf = pd.concat([xdf, dfTrain[['label']]], axis=1)
    plot(finaldf, model_name)

if __name__ == "__main__":
    main()
