import fileprocess
from textblob import TextBlob
from matplotlib import pyplot as plt


TRAINFILEPATH = "../fake-news/train.csv"
TESTFILEPATH = "../fake-news/test.csv"


def plt_polarity(dat, category):
    # Sentiment analysis on real and fake category
    dat.hist(grid=True, bins=50, rwidth=0.9, color='#607c8e')
    if category == "fake":
        plt.title('Distribution of Polarity of Fake News Sentiment')
    elif category == "real":
        plt.title('Distribution of Polarity of Real News Sentiment')
    else:  # all news
        plt.title('Distribution of Polarity of Sentiment')

    plt.ylabel('Count')
    plt.xlabel('Polarity')
    plt.grid(axis='y', alpha=0.75)


def main():
    # plot the sentiment analysis on Fake, real and all news
    data = fileprocess.read_files(TRAINFILEPATH,nolabel = False)
    data['polarity'] = data['text'].map(lambda x: TextBlob(x).sentiment.polarity)
    real = data.loc[data['label'] == 0] # select rows that are real news
    fake = data.loc[data['label'] == 1] # select rows that are real news
    plt.figure(1)
    plt_polarity(data['polarity'], "all")
    plt.figure(2)
    plt_polarity(fake['polarity'], "fake")
    plt.figure(3)
    plt_polarity(real['polarity'], "real")
    plt.show()

if __name__ == "__main__":
    main()
