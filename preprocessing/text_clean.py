import FileProcess as fp
import pandas as pd
import spacy
from string import digits


def remove_stop(data):
    spacy_nlp = spacy.load('en_core_web_sm')
    spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
    data['text'] = data['text'].apply(lambda x: 
    ' '.join([token.text for token in spacy_nlp(x) if not token.is_stop]))
    return data


def remove_num(data):
    # Credit goes to:
    # https://stackoverflow.com/questions/12851791/
    # removing-numbers-from-string/12851835
    data['text'] = data['text'].apply(lambda x:
        x.translate(str.maketrans('', '', digits)))
    data['title'] = data['title'].apply(lambda x:
        x.translate(str.maketrans('', '', digits)))
    return data

def main():
    data = fp.read_files(fp.TRAINFILEPATH,nolabel = False, sample=100)
    data = remove_stop(data)
    data = remove_num(data)
    filename = "train_clean.csv"
    data.to_csv(filename, encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
