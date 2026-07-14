from sklearn.feature_extraction.text import TfidfVectorizer


def train_tfidf(news_df):
    """
    Trains TF-IDF on title + summary of live-fetched articles.
    """
    news_df["text_for_model"] = (
        news_df["title"].fillna("") + " " + news_df["summary"].fillna("")
    )

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(news_df["text_for_model"])

    return tfidf, tfidf_matrix