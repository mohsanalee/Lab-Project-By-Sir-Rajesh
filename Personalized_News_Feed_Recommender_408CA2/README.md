# Personalized News Feed Recommender

A Streamlit web application that recommends live news articles to users based
on their registered interests and free-text search queries. Articles are
fetched in real time from NewsAPI and ranked using a TF-IDF + cosine
similarity content-based recommendation model.

## Features

- **User registration flow** — collects the user's name and interests
  (Sports, Entertainment, News) via `st.session_state`, no database required
- **Personalized "For You" feed** — automatically builds a ranked feed from
  the user's saved interests on login
- **Free-text search** — search any other topic beyond the saved interests
  (e.g. technology, business, health)
- **Content-based ranking** — TF-IDF vectorization + cosine similarity scores
  each article against the user's query/interests
- **Live data** — articles are fetched fresh from NewsAPI on every request,
  no static dataset

## Project Structure

```
├── app.py                     # Main Streamlit app (registration + home feed UI)
├── src/
│   ├── news_api.py            # Fetches live articles from NewsAPI
│   ├── model.py                # TF-IDF vectorizer training
│   └── recommender.py         # Builds user vector + ranks articles by similarity
├── .streamlit/
│   └── secrets.toml            # Stores NEWS_API_KEY (not committed to Git)
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Clone / download the project

Place all project files in a single folder, preserving the structure above.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get a NewsAPI key

1. Sign up for a free account at [newsapi.org](https://newsapi.org)
2. Copy your API key from the dashboard

### 4. Add your API key

Create a folder named `.streamlit` in the project root, and inside it a file
named `secrets.toml`:

```toml
NEWS_API_KEY = "your_actual_api_key_here"
```

> **Note:** Never commit `secrets.toml` to Git. Add it to `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Usage

1. **Register** — enter your name and select one or more interests
   (Sports / Entertainment / News)
2. **View your feed** — the home page automatically loads 6 articles ranked
   by relevance to your interests
3. **Search more** — use the search box to look up any other topic and get
   a ranked list of live results
4. **Edit interests** — click "Edit interests" at any time to re-register

## Known Limitations

- NewsAPI's free Developer plan is limited to 100 requests/day and returns
  articles from roughly the last month only
- User data (name, interests) is stored only in the browser session
  (`st.session_state`) and is not persisted between sessions or across devices
- No user authentication — registration is a lightweight preference form,
  not a secure login system

## Developers

| Name | Roll No |
|---|---|
| Mohsan Ali | 2k23/CSEE/40 |
| Sheeraz Ahmed | 2k23/CSEE/64 |
| Asad Niaz | 2k23/CSEE/18 |
