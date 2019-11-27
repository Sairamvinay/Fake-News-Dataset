import fileprocess as fp
import pandas as pd
import spacy
import sys
import numpy as np
import string
import re
import langid

# Usage: to clean train, do <python text_clean.py>

def remove_stop(data):
    # remove all stop words using spacy 
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    data['text'] = data['text'].apply(lambda x:
    ' '.join([token.text for token in spacy_nlp(x) if not token.is_stop]))
    return data


def remove_num(data):
    # remove all digits
    # Credit goes to:
    # https://stackoverflow.com/questions/12851791/
    # removing-numbers-from-string/12851835
    data['text'] = data['text'].apply(lambda x:
        x.translate(str.maketrans('', '', string.digits)))
    data['title'] = data['title'].apply(lambda x:
        x.translate(str.maketrans('', '', string.digits)))
    return data

def remove_punct_noneng(df):
    # punctuation removals on @, # and all the symbols
    punctuation_remove = string.punctuation
    punctuation_remove = punctuation_remove.replace('@', '')
    punctuation_remove = punctuation_remove.replace('#', '')
    df['text'] = df['text'].str.replace('[{}]'.format(punctuation_remove), '')
    list_to_remove = ["\r", "\n", "–", "“", "”", "…", "‘", "’", "•"]

    df['text'] = [re.sub(r"#\w+", "", str(x)) for x in df['text']]
    df['text'] = [re.sub(r"@\w+", "", str(x)) for x in df['text']]
    df['text'] = [re.sub("—", " ", str(x)) for x in df['text']] #replace - with space
    df["text"] = [re.sub('\s+', ' ', str(x)) for x in df["text"]]   #remove more than 2 consec spaces with just one space

    for elem in list_to_remove:
        df["text"] = df["text"].str.replace(elem, "")

    df["text"] = df["text"].str.lower()

    # remove all rows with foreign language characters
    for index, row in df.iterrows():
        text = row['text']
        # check for null text
        empty = text is np.nan or text != text
        if not empty:
            if len(text) >= 3:
                lang, _ = langid.classify(text)
                if lang != "en":
                    df.drop(index, inplace=True)
    return(df)


def main():
    #read in whole file for training
    data = fp.read_files(fp.TRAINFILEPATH,nolabel = False,sample = None)
    filename = "train_clean.csv"

    print("Doing data cleanup")

    data = remove_num(data)
    print("After removing numbers\n")

    data = remove_punct_noneng(data)
    print("After removing punctuations\n")

    data = remove_stop(data)
    print("After removing Stopwords\n")

    data.to_csv(filename, encoding='utf-8', index=False)



if __name__ == "__main__":
    main()
