import pandas as pd

TRAINFILEPATH = "../fake-news/train_clean.csv"

# Read cleaned (preprocessed) csv file
def read_clean_data(PATH):
    # baseline read data function to be used by other files 
    names = ["id","title","author","text","label"]

    df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
    df.dropna(how='any', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["text"] = df['text'].values.astype('U')
    return df
