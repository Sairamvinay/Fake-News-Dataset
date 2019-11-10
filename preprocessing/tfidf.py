from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df['text'].tolist())
    words = vectorizer.get_feature_names()
    score = tfidf.todense().tolist()
    return words, score
