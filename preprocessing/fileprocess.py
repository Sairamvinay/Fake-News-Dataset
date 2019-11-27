import pandas as pd

TRAINFILEPATH = "../fake-news/train_clean.csv"


def read_files(PATH,nolabel = False, sample=None):
	# file process will read the file and organize file into title as id, title, author and text
	names = []
	if nolabel == True:
		names = ["id","title","author","text"]

	else:
		names = ["id","title","author","text","label"]

	df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
	# drop all the NAN values
	df.dropna(how='any', inplace=True)
	# shuffle data and turn the data value to unicode
	df.reset_index(drop=True, inplace=True)

	df["text"] = df['text'].values.astype('U')

	if sample is None:
		return df
	else:
		return df.sample(n = sample,random_state = 999)


def main():
	# load, train and test data 
	dfTrain = read_files(TRAINFILEPATH,nolabel = False)

if __name__ == '__main__':
	main()
