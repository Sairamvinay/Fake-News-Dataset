import pandas as pd
import spacy
import csv
import time
from collections import defaultdict
nlp = spacy.load('en_core_web_sm')

# load in my data as data frame
df = pd.read_csv('../fake-news/train_clean.csv')

dfTrue = df[df['label']==0]
dfFalse = df[df['label']==1]

# find the size of the data frame

i = 1
# collect 5 features for now to do the analysis
POS = defaultdict(int)
TAG = defaultdict(int)
DEP = defaultdict(int)
ALPHA = defaultdict(int)
STOP = defaultdict(int)
with open("FalseDataFeature_dict.csv", "w", newline="") as f:
    writer = csv.writer(f, delimiter=',')
    for sen in dfFalse['title'].dropna():
        start_time = time.time()
        #print(sen)
        doc = nlp(sen)
        for token in doc:
            POS[token.pos_]+=1
            TAG[token.tag_]+=1
            DEP[token.dep_]+=1
            ALPHA[token.is_alpha]+=1
            STOP[token.is_stop]+=1
        second_time = time.time()
        print("time estimates to load ", i)
        print("th sample: %s seconds" % (second_time-start_time))
        i+=1
#row = [POS, TAG, DEP, ALPHA, STOP]
print(POS)
print(TAG)
print(DEP)
