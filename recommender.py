from sklearn.metrics.pairwise import cosine_similarity


def build_user_vector(tfidf, user_text):
    return tfidf.transform([user_text])


def recommend(news_df, tfidf_matrix, user_vector, interactions=None, top_n=5):
    """
    Ranks articles by cosine similarity to user interest vector.
    'interactions' kept as optional param for future use, unused for now.
    """
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    news_df = news_df.copy()
    news_df["score"] = scores

    results = news_df.sort_values("score", ascending=False).head(top_n)
    return results.reset_index(drop=True)