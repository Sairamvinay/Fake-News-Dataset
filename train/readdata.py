import pandas as pd

TRAINFILEPATH = "../fake-news/train_clean.csv"

def read_clean_data(PATH,nolabel = False, sample=None):
    # baseline read data function to be used by other files 
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
