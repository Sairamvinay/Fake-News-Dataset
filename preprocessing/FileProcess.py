import pandas as pd

TRAINFILEPATH = "../fake-news/train.csv"
TESTFILEPATH = "../fake-news/test.csv"

def read_files(PATH,nolabel = False, sample=None):
	
	names = []
	if nolabel == True:
		names = ["id","title","author","text"]

	else:
		names = ["id","title","author","text","label"]
	
	df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
	df.dropna(how='any', inplace=True)
	df.reset_index(drop=True, inplace=True)

	df = df['text'].values.astype('U')
	
	if sample is None:
		return df
	else:
		return df.sample(n = sample,random_state = 999)


def main():
	dfTrain = read_files(TRAINFILEPATH,nolabel = False)
	dfTest = read_files(TESTFILEPATH,nolabel = True)

if __name__ == '__main__':
	main()
