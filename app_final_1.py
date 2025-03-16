import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from moviedata_final import MovieData
import numpy as np

# Initialize the MovieData class
data = MovieData()

# Streamlit App UI
st.title("🎬 Movie Data Explorer")

# --- Movie Releases Over Time ---
st.header("📅 Movie Releases Over Time")
genre = st.selectbox("Select Genre (Optional)", [None, "Drama", "Comedy", "Action", "Romance", "Horror", "Thriller"])

release_df = data.releases(genre)
fig, ax = plt.subplots()
ax.bar(release_df["Year"], release_df["Movie_Count"], color='steelblue')
ax.set_xlabel("Year")
ax.set_ylabel("Number of Movies")
ax.set_title(f"Movies Released Per Year ({'All' if not genre else genre})")
st.pyplot(fig)

# --- Actor Births Over Time ---
st.header("👶 Actor Births Over Time")
age_unit = st.radio("Select Unit", ["Year", "Month"], index=0)
age_df = data.ages('Y' if age_unit == 'Year' else 'M')

fig, ax = plt.subplots()
ax.bar(age_df.iloc[:, 0], age_df.iloc[:, 1], color='darkorange')
ax.set_xlabel(age_unit)
ax.set_ylabel("Number of Births")
ax.set_title(f"Number of Actor Births by {age_unit}")
st.pyplot(fig)

# --- Movie Type Histogram ---
st.header("Movie Type Histogram")
N = st.number_input("Select Top N Movie Types", min_value=1, max_value=50, value=10, step=1)

movie_type_df = data.movie_type(N)
if "Movie_Type" in movie_type_df.columns and "Count" in movie_type_df.columns:
    fig, ax = plt.subplots()
    ax.bar(movie_type_df["Movie_Type"], movie_type_df["Count"], color='skyblue')
    ax.set_xlabel("Movie Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {N} Movie Types")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# --- Actor Count Histogram ---
st.header("Actor Count Histogram")
actor_count_df = data.actor_count()


if not actor_count_df.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(actor_count_df["Number of Actors"], actor_count_df["Movie Count"], color='lightcoral', width=0.8)
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Movie Count")
    ax.set_title("Actor Count Distribution")
    ax.set_yscale("log")
    st.pyplot(fig)
    st.markdown(
    "**Note:** The y-axis is displayed on a logarithmic scale to better visualize the distribution of movie counts across different numbers of actors."
)
else:
    st.warning("No actor count data available.")



# --- Actor Height Distribution ---
st.header("Actor Height Distribution")
gender = st.selectbox("Select Gender", ["All", "Male", "Female"])
min_height = st.number_input("Min Height (cm)", min_value=100, max_value=250, value=150)
max_height = st.number_input("Max Height (cm)", min_value=100, max_value=250, value=200)

height_df = data.actor_distributions(gender=gender.lower(), min_height=min_height, max_height=max_height)

if not height_df.empty:
    #st.write("Filtered Height Data:", height_df.head())  # Debugging step

    fig, ax = plt.subplots()
    ax.hist(height_df["Actor_height"], bins=20, color='mediumseagreen', edgecolor="black")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Height Distribution for {gender} Actors")
    st.pyplot(fig)
else:
    st.warning("No height data available for the selected filters.")

# --- Movie Genre Classification ---
st.header("🧠 Movie Genre Classification with LLM")

import random
from ollama import generate

# Function to get a random movie
def get_random_movie():
    random_movie = data.movie_df.sample(1).iloc[0]
    return random_movie["Movie_Name"], random_movie["Movie_genres"], random_movie.get("Movie_summary", "No summary available.")

if st.button("Shuffle"):
    title, genres, summary = get_random_movie()
    st.write(f"**Movie:** {title}")
    st.write(f"**Genres (Database):** {genres}")
    
    # Use Ollama model for classification
    prompt = f"""
    You are a movie classification assistant. Your task is to classify the following movie into genres 
    based on its summary. Prioritize choosing genres that are already present in the database.

    ONLY return a comma-separated list of genres from the predefined list below. Do NOT add extra text.

    Available Genres: Crime Fiction, Comedy film, Comedy-drama, Romantic comedy, Musical, Romance Film, 
    Comedy, Drama, Romantic drama, Action, Thriller, Science Fiction, Animation, Horror, Fantasy, 
    Documentary, Western, Biography, Mystery, Adventure, War, History, Sports.

    Database Genres for this movie: {genres}

    Movie Title: {title}
    Movie Summary: {summary}

    Genres:
    """
    response = generate("deepseek-r1:1.5b", prompt)
    st.write(f"**Genres (LLM Prediction):** {response['response'].strip()}")

   # Define the valid genres from the database
    valid_genres = {
        "Crime Fiction", "Comedy film", "Comedy-drama", "Romantic comedy", "Musical",
        "Romance Film", "Comedy", "Drama", "Romantic drama", "Action", "Thriller",
        "Science Fiction", "Animation", "Horror", "Fantasy", "Documentary", "Western",
        "Biography", "Mystery", "Adventure", "War", "History", "Sports"
    }

    # Extract response safely
    predicted_genres = response.get("response", "").strip() if isinstance(response, dict) else str(response).strip()

    # Convert predicted genres to a set, filtering only valid ones
    predicted_genres = set(
        genre.strip() for genre in predicted_genres.split(',')
        if genre.strip() in valid_genres
    )

    # Convert actual genres from the database
    actual_genres = set(genres.split(',')) if isinstance(genres, str) else set()

    # Compare model classification with database genres
    match = predicted_genres.intersection(actual_genres)

    if match:
        st.success("✅ LLM classification matches existing genres!")
    else:
        st.error("❌ LLM classification differs from database genres.")