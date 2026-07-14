import streamlit as st

from src.news_api import fetch_news
from src.model import train_tfidf
from src.recommender import build_user_vector, recommend

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Personalized News Feed Recommender",
    page_icon="📰",
    layout="centered"
)

if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- GLOBAL STYLE ----------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

    :root {
        --ink: #16213D;
        --ink-soft: #414A63;
        --paper: #FBF9F5;
        --paper-raised: #FFFFFF;
        --accent-deep: #96721C;
        --accent-wash: #FBF0D9;
        --line: #E4E0D4;
        --muted: #7B7E8C;
    }

    .stApp { background-color: var(--paper); }
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Source Serif 4', Georgia, serif !important; }

    /* Masthead */
    .masthead {
        border-bottom: 3px solid var(--ink);
        padding-bottom: 14px;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        flex-wrap: wrap;
        gap: 10px;
    }
    .masthead h1 {
        font-size: 1.9rem;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .masthead h1 span { color: var(--accent-deep); }
    .masthead .dateline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--muted);
    }
    .masthead-rule { height: 1px; background: var(--ink); opacity: 0.15; margin-bottom: 30px; }

    /* Inputs */
    div[data-testid="stTextInput"] input {
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
        color: var(--ink);
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent-deep);
        box-shadow: 0 0 0 3px var(--accent-wash);
    }

  /* Checkboxes styled as pills */
    div[data-testid="stCheckbox"] label {
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 10px 18px;
        display: flex;
        align-items: center;
        gap: 10px;
        cursor: pointer;
    }
    div[data-testid="stCheckbox"] p {
        color: var(--ink) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
        opacity: 1 !important;
    }
    div[data-testid="stCheckbox"] label:has(input:checked) {
        border-color: var(--accent-deep);
        background-color: var(--accent-wash);
    }
    div[data-testid="stCheckbox"] label:has(input:checked) p {
        color: var(--accent-deep) !important;
    }
    /* Fix invisible markdown text (bold labels + headers) */
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stMarkdownContainer"] h1,
    div[data-testid="stMarkdownContainer"] h2,
    div[data-testid="stMarkdownContainer"] h3 {
        color: var(--ink) !important;
        opacity: 1 !important;
    }

    div[data-testid="stMarkdownContainer"] h3 {
        font-family: 'Source Serif 4', Georgia, serif !important;
        font-weight: 600 !important;
        margin-top: 6px !important;
    }

    /* Buttons */
    .stButton>button, .stFormSubmitButton>button {
        background-color: var(--ink);
        color: var(--paper) !important;
        border: none;
        border-radius: 8px;
        padding: 11px 26px;
        font-weight: 600;
    }
    .stButton>button:hover, .stFormSubmitButton>button:hover { background-color: #263258; }
    /* Force all button text (including nested p tags) to white */
    .stButton>button,
    .stFormSubmitButton>button {
        background-color: var(--ink);
        color: var(--paper) !important;
        border: none;
        border-radius: 8px;
        padding: 11px 26px;
        font-weight: 600;
    }
    .stButton>button p,
    .stFormSubmitButton>button p,
    .stButton>button span,
    .stFormSubmitButton>button span,
    .stButton>button div,
    .stFormSubmitButton>button div {
        color: var(--paper) !important;
        opacity: 1 !important;
    }
    .stButton>button:hover,
    .stFormSubmitButton>button:hover {
        background-color: #263258;
    }
    .stButton>button:hover p,
    .stFormSubmitButton>button:hover p {
        color: var(--paper) !important;
    }

    /* Registration card look around the form */
   div[data-testid="stTextInput"] input {
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 12px 14px;
        color: var(--ink);
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: var(--muted);
        opacity: 1;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: var(--accent-deep);
        box-shadow: 0 0 0 3px var(--accent-wash);
    }

    /* Post cards */
    .news-card {
        background-color: var(--paper-raised);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 16px;
    }
    .news-meta { font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
    .news-title { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
    .news-summary { font-size: 14px; color: var(--ink-soft); margin-bottom: 10px; }
    .news-link a { color: var(--accent-deep); font-weight: 600; font-size: 13px; text-decoration: none; }
    .news-link a:hover { text-decoration: underline; }

    .tag {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; text-transform: uppercase;
        background-color: var(--accent-wash); color: var(--accent-deep);
        padding: 4px 12px; border-radius: 999px; margin-right: 6px;
    }

    .site-footer {
        border-top: 3px solid var(--ink);
        margin-top: 50px; padding-top: 24px; text-align: center;
    }
    .site-footer .about { font-size: 13px; color: var(--ink-soft); max-width: 48ch; margin: 0 auto 18px; }
    .site-footer .credits {
        font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: var(--muted);
        border-top: 1px dashed var(--line); padding-top: 16px; display: inline-block;
    }
    .site-footer .credits strong { display: block; color: var(--ink); font-size: 10px; text-transform: uppercase; margin-bottom: 6px; }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- MASTHEAD HEADER  ----------------
import datetime
today = datetime.date.today().strftime("%A, %B %d, %Y")

st.markdown(
    f"""
    <div class="masthead">
        <h1>Personalized News <span>Feed Recommender</span></h1>
    </div>
    <div class="masthead-rule"></div>
    """,
    unsafe_allow_html=True
)

# STEP 1 — REGISTRATION

def show_registration():
    st.markdown(
        "<p style='color:#7B7E8C; margin-top:-10px; margin-bottom:24px;'>"
        "Tell us who you are and what you follow — we'll build your first feed from it."
        "</p>",
        unsafe_allow_html=True
    )

    with st.form("registration_form"):
        name = st.text_input("Your name", placeholder=  "Enter your name here to register")

        st.write("**Choose your interests**")
        col1, col2, col3 = st.columns(3)
        with col1:
            sports = st.checkbox("Sports")
        with col2:
            entertainment = st.checkbox("Entertainment")
        with col3:
            news_topic = st.checkbox("News")

        submitted = st.form_submit_button("Continue to my feed →")

        if submitted:
            interests = []
            if sports: interests.append("Sports")
            if entertainment: interests.append("Entertainment")
            if news_topic: interests.append("News")

            if not name.strip():
                st.error("Please enter your name.")
            elif not interests:
                st.error("Please choose at least one interest.")
            else:
                st.session_state.user = {"name": name.strip(), "interests": interests}
                st.rerun()


# STEP 2 — HOME / FEED
def render_post_card(row):
    st.markdown(
        f"""
        <div class="news-card">
            <div class="news-meta">{row.get('source', 'N/A')} · Score: {round(row['score'], 3)}</div>
            <div class="news-title">{row['title']}</div>
            <div class="news-summary">{row['summary']}</div>
            <div class="news-link"><a href="{row['url']}" target="_blank">Read full article →</a></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_ranked_feed(query, top_n):
    news = fetch_news(query, page_size=25)
    if news.empty:
        return news
    tfidf, tfidf_matrix = train_tfidf(news)
    user_vector = build_user_vector(tfidf, query)
    return recommend(news, tfidf_matrix, user_vector, top_n=top_n)


def show_home():
    user = st.session_state.user

    top_col, edit_col = st.columns([4, 1])
    with top_col:
        st.subheader(f"Welcome back, {user['name']}")
        st.markdown(
            " ".join(f"<span class='tag'>{i}</span>" for i in user["interests"]),
            unsafe_allow_html=True
        )
    with edit_col:
        st.write("")
        if st.button("Edit interests"):
            st.session_state.user = None
            st.session_state.pop("for_you_feed", None)
            st.rerun()

    st.write("")

    if "for_you_feed" not in st.session_state:
        with st.spinner("Building your feed..."):
            interest_query = " OR ".join(user["interests"])
            st.session_state.for_you_feed = get_ranked_feed(interest_query, top_n=6)

    st.markdown("### For you")
    feed = st.session_state.for_you_feed
    if feed.empty:
        st.info("Couldn't load your feed right now. Try searching below.")
    else:
        for _, row in feed.iterrows():
            render_post_card(row)

    st.markdown("---")
    st.markdown("### Search more")
    search_query = st.text_input("Search any topic", placeholder="e.g. technology, business, health...")
    top_n = st.slider("Number of results", 1, 10, 5)

    if st.button("Search"):
        if not search_query.strip():
            st.warning("Enter a topic to search.")
        else:
            with st.spinner(f"Fetching news for '{search_query}'..."):
                results = get_ranked_feed(search_query, top_n=top_n)
            if results.empty:
                st.warning("No articles found. Try different keywords.")
            else:
                st.markdown(f"#### Results for \"{search_query}\"")
                for _, row in results.iterrows():
                    render_post_card(row)


# ---------------- ROUTER ----------------
if st.session_state.user is None:
    show_registration()
else:
    show_home()

# ---------------- FOOTER ----------------
st.markdown(
    """
    <div class="site-footer">
        <p class="about">
            Personalized News Feed Recommender fetches live news via NewsAPI and uses
            TF-IDF based content similarity to rank articles matching your interests and searches.
        </p>
        <div class="credits">
            <strong>Developers</strong>
            Mohsan Ali &nbsp;|&nbsp; Roll No: 2k23/CSEE/40<br>
            Sheeraz Ahmed &nbsp;|&nbsp; Roll No: 2k23/CSEE/64<br>
            Asad Niaz &nbsp;|&nbsp; Roll No: 2k23/CSEE/18
        </div>
    </div>
    """,
    unsafe_allow_html=True
)