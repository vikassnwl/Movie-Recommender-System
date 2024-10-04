import streamlit as st
import numpy as np
import joblib
import requests
import gzip


def load_similarity():
    with gzip.open("artifacts/similarity.pkl.gz", "rb") as f:
        similarity_matrix = joblib.load(f)
    return similarity_matrix


def fetch_poster(movie_id):
    api_key = "8265bd1679663a7ea12ac168da84d2e8"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"
    response = requests.get(url)
    if response.status_code == 404:
        st.write(url)
        raise Exception("The resource you requested could not be found.")
    data = response.json()
    return "https://image.tmdb.org/t/p/w500" + data["poster_path"]


new_df = joblib.load("artifacts/movies.pkl")
similarity = load_similarity()


def recommend(movie):
    movie_index = np.where(new_df["title"] == movie)[0][0]
    distances = similarity[movie_index]
    movies_list = sorted(enumerate(distances), key=lambda x: x[1], reverse=True)[1:6]
    recommended_movies = []
    recommended_movies_posters = []
    for i in movies_list:
        movie_id = new_df.iloc[i[0]].movie_id
        recommended_movies.append(new_df.iloc[i[0]].title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters


st.title("Movie Recommender System")
selected_movie_name = st.selectbox("Select movie name", new_df["title"])
if st.button("Recommend"):
    names, posters = recommend(selected_movie_name)
    cols = st.columns(5)
    for i, col in enumerate(cols):
        with col:
            st.text(names[i])
            st.image(posters[i])
