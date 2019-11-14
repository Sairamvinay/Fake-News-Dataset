import pandas as pd
from gensim.models import Word2Vec
from matplotlib import pyplot
from sklearn.decomposition import PCA

### TO USE:
############ YOU NEED TO UNCOMMENT A LINE IN word2Vec TO CREATE A NEW MODEL
############ YOU NEED TO UNCOMMENT THE CODE AT THE BOTTOM

TRAINFILEPATH = "../fake-news/train_clean.csv"
TESTFILEPATH = "../fake-news/test_clean.csv"

def word2Vec(text, data_file):
    if (data_file == "train"):
        save_file = 'train_word2vec_model.bin'
    elif (data_file == "test"):
        save_file = 'test_word2vec_model.bin'
    else:
        print("ERROR: word2Vec did not accept args %s %s", text, data_file)
        return
############### REMOVE TO CREATE A NEW MODEL ##################
    return Word2Vec.load(save_file)
############### REMOVE TO CREATE A NEW MODEL ##################
    # train model
    model = Word2Vec(text, min_count=10)

    # summarize vocabulary
    words = list(model.wv.vocab)

    # save model
    model.save(save_file)
    
    return model

def graph(model):
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
        pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    import pandas as pd

def read_files(PATH,nolabel = False, sample=None):

    names = []
    if nolabel == True:
        names = ["id","title","author","text"]

    else:
        names = ["id","title","author","text","label"]

    df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df["text"] = df['text'].values.astype('U')
    
    if sample is None:
        return df
    else:
        return df.sample(n = sample,random_state = 999)


def main(argv):
    if (argv == "train"):
        file_path = TRAINFILEPATH
        nolabel = False
    elif (argv == "test"):
        file_path = TESTFILEPATH
        nolabel = True
    else:
        print('Please enter "train" or "test"')
        return

    df = read_files(file_path,nolabel=nolabel)

    lines_length = len(df.values)
    text = [df["text"].values[i].split(" ") for i in range(lines_length)]
    model = word2Vec(text, argv)
    print(model)
    
''' UNCOMMET TO RUN '''
# print("Training")
# main("train")
# print("\n\n\nTesting")
# main("test")
