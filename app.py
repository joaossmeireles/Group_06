import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from moviedata import MovieData

# Initialize the MovieData class
data = MovieData()

# Streamlit App UI
st.title("🎬 Movie Data Explorer")

# --- Movie Type Histogram ---
st.header("Movie Type Histogram")
N = st.number_input("Select Top N Movie Types", min_value=1, max_value=50, value=10, step=1)

try:
    movie_type_df = data.movie_type(N)
    fig, ax = plt.subplots()
    ax.bar(movie_type_df["Movie_Type"], movie_type_df["Count"], color='skyblue')
    ax.set_xlabel("Movie Type")
    ax.set_ylabel("Count")
    ax.set_title(f"Top {N} Movie Types")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error: {e}")
# --- Actor Count Histogram ---
st.header("Actor Count Histogram")
try:
    actor_count_df = data.actor_count()
    fig, ax = plt.subplots()
    ax.bar(actor_count_df["Number of Actors"], actor_count_df["Movie Count"], color='lightcoral')
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Movie Count")
    ax.set_title("Actor Count Distribution")
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error: {e}")

# --- Actor Height Distribution ---
st.header("Actor Height Distribution")

gender_options = ["All", "Male", "Female"]  # Adjust based on actual dataset values
gender = st.selectbox("Select Gender", gender_options)
min_height = st.number_input("Min Height (cm)", min_value=100, max_value=250, value=150, step=1)
max_height = st.number_input("Max Height (cm)", min_value=100, max_value=250, value=200, step=1)

plot_distribution = st.checkbox("Plot Distribution", value=True)

try:
    height_df = data.actor_distributions(gender=gender, min_height=min_height, max_height=max_height, plot=False)
    st.write(height_df)  # Display filtered data
    
    if plot_distribution:
        fig, ax = plt.subplots()
        ax.hist(height_df["Height"], bins=20, edgecolor="black", color='mediumseagreen')
        ax.set_xlabel("Height (cm)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Height Distribution for {gender} Actors")
        st.pyplot(fig)
except Exception as e:
    st.error(f"Error: {e}")