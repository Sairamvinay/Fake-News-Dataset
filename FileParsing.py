import pandas as pd

TRAINFILEPATH = "fake-news/train.csv"
TESTFILEPATH = "fake-news/test.csv"

def read_files(PATH,nolabel = False):
	
	names = []
	if nolabel == True:
		names = ["id","title","author","text"]

	else:
		names = ["id","title","author","text","label"]
	
	df = pd.read_csv(PATH,sep = ",",names= names,header = 0)
	return df


def main():
	dfTrain = read_files(TRAINFILEPATH,nolabel = False)
	dfTest = read_files(TESTFILEPATH,nolabel = True)

if __name__ == '__main__':
	main()
