import pandas as pd

TRAINFILEPATH = "../fake-news/train_clean.csv"


# Read the csv file
def read_files(PATH):
	names = ["id","title","author","text","label"]

	df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
	# drop all the NAN values
	df.dropna(how='any', inplace=True)
	# shuffle data and turn the data value to unicode
	df.reset_index(drop=True, inplace=True)

	df["text"] = df['text'].values.astype('U')
	return df

