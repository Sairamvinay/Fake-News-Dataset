import spacy
from collections import defaultdict
from fileprocess import read_files,TRAINFILEPATH
import pickle
import numpy as np
import matplotlib.pyplot as plt
NLP = spacy.load("en_core_web_sm")

def find_rat_type(label):

	if label == 0:
		return "REAL"

	else:
		return "FAKE"

def update_dict(dictn,tags,POS_TAGS):
	# count POS frequency on each token
	for pos in POS_TAGS:
		dictn[pos] += tags.count(pos)

	return dictn

def check_tag_distr(text,label,POS_TAGS):
	# dictionary for real and fake TAGS
	count_real = defaultdict(int)
	count_fake = defaultdict(int)
	ALLTAGS = []
	ALL_TEXT_TAGS = []


	for i in range(len(label)):

		if (i%1000 == 0):
			print(i,"th line")

		text_curr = text[i]
		label_curr = label[i]

		type_rat = find_rat_type(label_curr)
		# generate all POS tags
		doc = NLP(text_curr)

		tags = [tok.pos_ for tok in doc]

		ALL_TEXT_TAGS.append(tags)
		ALLTAGS.extend(tags)

		# record frequency on both fake and real
		if type_rat == "FAKE":
			count_fake = update_dict(count_fake,tags,POS_TAGS)

		else:
			count_real = update_dict(count_real,tags,POS_TAGS)



	print("Set of all different tags: ",set(ALLTAGS))
	return (count_real,count_fake)



def save_tag_prop(df,filename,POS_TAGS):
	# save the POS distribution for both real and fake news data
	text = df["text"].tolist()
	label = df["label"].tolist()

	count_real,count_fake = check_tag_distr(text,label,POS_TAGS = POS_TAGS)

	print(count_real," is the real dictionary")
	print(count_fake," is the fake dictionary")

	file1 = open("RealPOS.pkl","wb")
	file2 = open("FakePOS.pkl","wb")
	pickle.dump(count_real,file1)
	pickle.dump(count_fake,file2)
	file1.close()
	file2.close()



def tag_distr(df,filename):
	# POS tag distribution on "ADV","ADJ","NOUN","VERB","PROPN"
	POS_TAGS = ["ADV","ADJ","NOUN","VERB","PROPN"]

	save_tag_prop(df,filename,POS_TAGS)

def graphing(T = "Fake"):
	# graphing POS distribution
	file1 = open("RealPOS.pkl",'rb')
	file2 = open("FakePOS.pkl",'rb')

	count_real = pickle.load(file1)
	count_fake = pickle.load(file2)
	POS_TAGS = ["ADV","ADJ","NOUN","VERB","PROPN"]
	getpropreal = []
	getpropfake = []
	for tag in POS_TAGS:
		cr = count_real[tag]
		cf = count_fake[tag]

		sum_all = (cr + cf) * 0.01
		getpropreal.append(cr/sum_all)
		getpropfake.append(cf/sum_all)

	x_pos = np.arange(len(POS_TAGS))
	Y = []
	title = str()
	color = str()
	if T == "Fake":
		Y = getpropfake
		title = "Fake News POS tag distribution"
		color = "red"

	else:
		Y = getpropreal
		title = "Real News POS tag distribution"
		color = "blue"

	plt.bar(x_pos, Y, align='center', alpha=0.5,color = color)
	plt.xticks(x_pos, POS_TAGS)
	plt.ylabel("Proportion of news %")
	plt.title(title)
	plt.show()

dfTrain = read_files(TRAINFILEPATH,nolabel = False,sample = None)
tag_distr(dfTrain,"Train")
graphing(T = "Fake")
graphing(T = "Real")
