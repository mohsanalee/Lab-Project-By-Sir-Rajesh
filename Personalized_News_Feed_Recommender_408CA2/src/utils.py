def explain(row):
    return f"""
    Recommended because:
    - Category: {row['category']}
    - High relevance to your interests
    - Popular among users
    - Score: {round(row['score'], 2)}
    """