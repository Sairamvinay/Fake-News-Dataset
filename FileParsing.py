import pandas as pd
import spacy
import csv

	
	


spacy_nlp = spacy.load('en_core_web_sm')
TRAINFILEPATH = "fake-news/train.csv"
TESTFILEPATH = "fake-news/test.csv"

def read_files(PATH,nolabel = False,sample=None):
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
	

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

#print('Number of stop words: %d' % len(spacy_stopwords))
#print('First ten stop words: %s' % list(spacy_stopwords)[:10])

dfTrain = read_files(TRAINFILEPATH,nolabel = False, sample=10)
dfTest = read_files(TESTFILEPATH,nolabel = True)
textTrain = dfTrain["text"]




# with open('clesdad.csv', 'w') as f:
	#thewriter = csv.writer(f)
	# for i in textTrain:
file_name = "abc.csv"
dfTrain["no_stop"] = textTrain.map(lambda x: ' '.join([token.text for token in spacy_nlp(x) if not token.is_stop]))
dfTrain.to_csv(file_name, sep='\t', encoding='utf-8')
		# doc = spacy_nlp(i)
		# tokens = [token.text for token in doc if not token.is_stop]	
		# #tokens.remove("\n")
		# tokens = ' '.join(tokens)

		# print(tokens)
		#f.write(' '.join(tokens) + "\n")
		

#print('Original Article: %s' % (doc))
#print()
#print(tokens)


#print(textTrain[0])
#print(dfTrain)
#print(dfTest)

