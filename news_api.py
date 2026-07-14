import requests
import streamlit as st
import pandas as pd

EVERYTHING_URL = "https://newsapi.org/v2/everything"


def fetch_news(query, page_size=20):
    """
    Fetch live news for ANY keyword/topic the user searches.
    """
    api_key = st.secrets["NEWS_API_KEY"]

    if not query or not query.strip():
        query = "news"  # fallback if user searches with empty input

    params = {
        "q": query.strip(),
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(EVERYTHING_URL, params=params)

    if response.status_code != 200:
        st.error(f"NewsAPI error {response.status_code}: {response.text}")
        return pd.DataFrame()

    articles = response.json().get("articles", [])

    rows = []
    for a in articles:
        title = a.get("title") or ""
        summary = a.get("description") or ""
        content = a.get("content") or ""

        if not title or title == "[Removed]":
            continue

        rows.append({
            "title": title,
            "summary": summary,
            "content": content,
            "url": a.get("url", ""),
            "source": a.get("source", {}).get("name", ""),
            "publishedAt": a.get("publishedAt", ""),
        })

    return pd.DataFrame(rows)