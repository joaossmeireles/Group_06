import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from moviedata_test1 import MovieData
import numpy as np

# Initialize the MovieData class
data = MovieData()

# Streamlit App UI
st.title("🎬 Movie Data Explorer")

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

# 🔍 Debugging output
st.write("DEBUG: Raw character DataFrame (first 5 rows):", data.character_df.head())
st.write("DEBUG: actor_count_df (first 5 rows):", actor_count_df.head())

if not actor_count_df.empty:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(actor_count_df["Number of Actors"], actor_count_df["Movie Count"], color='lightcoral', width=0.8)
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Movie Count")
    ax.set_title("Actor Count Distribution")
    ax.set_yscale("log")
    st.pyplot(fig)
else:
    st.warning("No actor count data available.")



# --- Actor Height Distribution ---
st.header("Actor Height Distribution")
gender = st.selectbox("Select Gender", ["All", "Male", "Female"])
min_height = st.number_input("Min Height (cm)", min_value=100, max_value=250, value=150)
max_height = st.number_input("Max Height (cm)", min_value=100, max_value=250, value=200)

height_df = data.actor_distributions(gender=gender.lower(), min_height=min_height, max_height=max_height)

if not height_df.empty:
    st.write("Filtered Height Data:", height_df.head())  # Debugging step

    fig, ax = plt.subplots()
    ax.hist(height_df["Actor_height"], bins=20, color='mediumseagreen', edgecolor="black")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Height Distribution for {gender} Actors")
    st.pyplot(fig)
else:
    st.warning("No height data available for the selected filters.")
