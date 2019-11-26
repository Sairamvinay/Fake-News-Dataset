from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize the data into TFIDF
# generate a list of TFIDF scores and a list of words 
def tfidf(df):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(df['text'].tolist())
    words = vectorizer.get_feature_names()
    score = tfidf.todense()
    return words, score
