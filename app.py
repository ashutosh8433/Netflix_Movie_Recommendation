import streamlit as st
import pandas as pd
import requests

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="Netflix Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)


# ==============================
# TITLE
# ==============================

st.title("🎬 Netflix Movie Recommendation System")

st.write(
    "Select a Netflix movie and get similar movie recommendations."
)


# ==============================
# LOAD DATASET
# ==============================

@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    return df


df = load_data()


# ==============================
# KEEP ONLY MOVIES
# ==============================

movies = df[df["type"] == "Movie"].copy()


# ==============================
# HANDLE MISSING VALUES
# ==============================

columns = [
    "director",
    "cast",
    "listed_in",
    "description"
]

for column in columns:
    movies[column] = movies[column].fillna("")


# ==============================
# COMBINE FEATURES
# ==============================

movies["combined_features"] = (
    movies["director"] + " " +
    movies["cast"] + " " +
    movies["listed_in"] + " " +
    movies["description"]
)


# ==============================
# TF-IDF
# ==============================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)


# ==============================
# COSINE SIMILARITY
# ==============================

cosine_sim = cosine_similarity(tfidf_matrix)


# ==============================
# RESET INDEX
# ==============================

movies = movies.reset_index(drop=True)


# ==============================
# CREATE MOVIE INDEX
# ==============================

indices = pd.Series(
    movies.index,
    index=movies["title"].str.lower()
).drop_duplicates()


# ==============================
# RECOMMENDATION FUNCTION
# ==============================

def recommend_movies(movie_title, number=10):

    movie_title = movie_title.lower().strip()

    if movie_title not in indices:
        return None

    movie_index = indices[movie_title]

    similarity_scores = list(
        enumerate(cosine_sim[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the selected movie itself
    similarity_scores = similarity_scores[
        1:number + 1
    ]

    movie_indices = [
        item[0]
        for item in similarity_scores
    ]

    recommendations = movies.iloc[
        movie_indices
    ].copy()

    recommendations["similarity_score"] = [
        round(item[1] * 100, 2)
        for item in similarity_scores
    ]

    return recommendations


# ==============================
# MOVIE POSTER / IMAGE
# ==============================

# Optional: add your TMDB API key in Streamlit secrets as:
# TMDB_API_KEY = "your_api_key_here"
# If no key is provided, a clean placeholder image is shown.
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")

@st.cache_data(show_spinner=False)
def get_movie_poster(movie_title):
    if not TMDB_API_KEY:
        return f"https://placehold.co/300x450/222222/ffffff?text={requests.utils.quote(movie_title)}"

    try:
        response = requests.get(
            "https://api.themoviedb.org/3/search/movie",
            params={
                "api_key": TMDB_API_KEY,
                "query": movie_title,
                "language": "en-US",
                "include_adult": "false"
            },
            timeout=10
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        if results and results[0].get("poster_path"):
            return "https://image.tmdb.org/t/p/w500" + results[0]["poster_path"]

    except requests.RequestException:
        pass

    return f"https://placehold.co/300x450/222222/ffffff?text={requests.utils.quote(movie_title)}"


# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("⚙️ Settings")

if TMDB_API_KEY:
    st.sidebar.success("🎞️ Movie posters enabled")
else:
    st.sidebar.info(
        "🎞️ Add a TMDB API key to show real movie posters. "
        "Without it, placeholders will be shown."
    )


number = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=15,
    value=10
)


# ==============================
# MOVIE SELECTION
# ==============================

movie_titles = sorted(
    movies["title"].dropna().unique()
)

selected_movie = st.selectbox(
    "🎥 Select a movie",
    movie_titles
)


# ==============================
# RECOMMEND BUTTON
# ==============================

if st.button("🔍 Recommend Movies"):

    recommendations = recommend_movies(
        selected_movie,
        number
    )

    if recommendations is None:

        st.error("Movie not found.")

    else:

        st.subheader(
            f"Movies similar to {selected_movie}"
        )

        for _, row in recommendations.iterrows():

            st.markdown("---")

            col1, col2, col3 = st.columns([1.3, 4, 1])

            with col1:
                poster_url = get_movie_poster(row["title"])
                st.image(
                    poster_url,
                    caption=row["title"],
                    use_container_width=True
                )

            with col2:

                st.markdown(
                    f"### 🎬 {row['title']}"
                )

                st.write(
                    f"**Genre:** {row['listed_in']}"
                )

                st.write(
                    f"**Release Year:** {row['release_year']}"
                )

                st.write(
                    f"**Rating:** {row['rating']}"
                )

                if row["director"]:

                    st.write(
                        f"**Director:** {row['director']}"
                    )

                st.write(
                    f"**Description:** "
                    f"{row['description']}"
                )

            with col3:

                st.metric(
                    "Similarity",
                    f"{row['similarity_score']}%"
                )


# ==============================
# DATASET INFORMATION
# ==============================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "📊 Dataset Information"
)

st.sidebar.write(
    f"Total Netflix Titles: {len(df)}"
)

st.sidebar.write(
    f"Total Movies: {len(movies)}"
)

st.sidebar.write(
    f"Total TV Shows: "
    f"{len(df[df['type'] == 'TV Show'])}"
)