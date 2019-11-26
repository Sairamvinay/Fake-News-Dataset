import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA
from fileprocess import read_files
### TO USE:
############ YOU NEED TO UNCOMMENT A LINE IN word2Vec TO CREATE A NEW MODEL
############ YOU NEED TO UNCOMMENT THE CODE AT THE BOTTOM

TRAINFILEPATH = "../fake-news/train_clean.csv"
save_file = '../fake-news/train_word2vec_model.bin'

def word2Vec(text, data_set):
    if (data_set == "train"):
        save_file = '../fake-news/train_word2vec_model.bin'
    else:
        print("ERROR: word2Vec did not accept args %s %s", text, data_set)
        return
############### REMOVE TO CREATE A NEW MODEL ##################
    return Word2Vec.load(save_file)
############### REMOVE TO CREATE A NEW MODEL ##################

def word2vecModel(text):
    # # train model
    min_count = sum(len(words) for words in text) / len(text)
    print(min_count)
    model = Word2Vec(text, min_count=min_count, workers=10)
    # # summarize vocabulary
    words = list(model.wv.vocab)
    # # save model
    model.save(save_file)
    return model

def graph(model):
    # Use PCA to reduce features to 2
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    import pandas as pd

def main(argv):
    file_path = TRAINFILEPATH
    nolabel = False
    df = read_files(file_path,nolabel=nolabel)
    lines_length = len(df.values)
    text = [df["text"].values[i].split(" ") for i in range(lines_length)]
    if (argv == "train"):
        model = word2vecModel(text)
    else:
        model = word2Vec(text, argv)
    print(model)

if __name__ == "__main__":
    main("train_else")

''' UNCOMMET TO RUN '''
#print("Training")
#main("train_else")
