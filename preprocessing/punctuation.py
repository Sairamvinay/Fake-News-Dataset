import numpy as np
import string
import re
import langid

from preprocessing.fileprocess import read_files, TRAINFILEPATH, TESTFILEPATH

dfTrain = read_files(TRAINFILEPATH, nolabel=False)
dfTest = read_files(TESTFILEPATH, nolabel=True)


def cleanup_data(df):
    punctuation_remove = string.punctuation
    punctuation_remove = punctuation_remove.replace('@', '')
    punctuation_remove = punctuation_remove.replace('#', '')
    df['text'] = df['text'].str.replace('[{}]'.format(punctuation_remove), '')
    list_to_remove = ["\r", "\n", "–", "“", "”", "…", "‘", "’", "•"]

    df['text'] = [re.sub(r"#\w+", "", str(x)) for x in df['text']]
    df['text'] = [re.sub(r"@\w+", "", str(x)) for x in df['text']]
    df['text'] = [re.sub("—", " ", str(x)) for x in df['text']]

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
    print(df)


cleanup_data(dfTrain)
cleanup_data(dfTest)

dfTrain.to_csv(TRAINFILEPATH)
dfTest.to_csv(TESTFILEPATH)
