import pandas as pd
import spacy
spacy_nlp = spacy.load('en_core_web_sm')
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

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#print('Number of stop words: %d' % len(spacy_stopwords))
#print('First ten stop words: %s' % list(spacy_stopwords)[:10])

dfTrain = read_files(TRAINFILEPATH,nolabel = False)
dfTest = read_files(TESTFILEPATH,nolabel = True)
textTrain = dfTrain["text"]

doc = spacy_nlp(textTrain[0])
tokens = [token.text for token in doc if not token.is_stop]
print('Original Article: %s' % (doc))
print()
print(tokens)

#print(textTrain[0])
#print(dfTrain)
#print(dfTest)

