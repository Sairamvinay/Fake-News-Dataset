import FileProcess
from textblob import TextBlob
from matplotlib import pyplot as plt


TRAINFILEPATH = "../fake-news/train.csv"
TESTFILEPATH = "../fake-news/test.csv"







def main():
    data = FileProcess.read_files(TRAINFILEPATH,nolabel = False, sample=500)
    data['polarity'] = data['text'].map(lambda x: TextBlob(x).sentiment.polarity)
    data.hist(column="polarity", bins=30)
    plt.ylabel('count')
    plt.xlabel('polarity')
    plt.show()


if __name__ == "__main__":
    main()