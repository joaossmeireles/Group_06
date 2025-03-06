import os
import tarfile
import requests
import pandas as pd
from pathlib import Path
from pydantic import BaseModel, ConfigDict
from typing import Optional

class MovieData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  
    movie_df: Optional[pd.DataFrame] = None
    character_df: Optional[pd.DataFrame] = None
    data_url: str = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    download_dir: Path = Path("downloads")
    extracted_dir: Path = download_dir / "MovieSummaries"

    def __init__(self):
        """Runs the download, extraction, and data loading process."""
        super().__init__()
        self.download_dir.mkdir(exist_ok=True)
        self._download_data()
        self._extract_data()
        self._load_movie_data()
        self._load_character_data()


    def _download_data(self):
        """Downloads the dataset if it doesn't already exist."""
        file_path = self.download_dir / "MovieSummaries.tar.gz"
        if not file_path.exists():
            print("Downloading dataset...")
            response = requests.get(self.data_url, stream=True)
            response.raise_for_status()  # Raise an error if the request fails
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print("Download complete.")
        else:
            print("Dataset already exists. Skipping download.")

    def _extract_data(self):
        """Extracts the dataset if not already extracted."""
        file_path = self.download_dir / "MovieSummaries.tar.gz"
        if not self.extracted_dir.exists():
            print("Extracting dataset...")
            with tarfile.open(file_path, "r:gz") as tar_ref:
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not os.path.commonprefix([os.path.abspath(path), os.path.abspath(member_path)]) == os.path.abspath(path):
                            raise Exception("Attempted Path Traversal in Tar File")
                    tar.extractall(path, members, numeric_owner=numeric_owner) 

                safe_extract(tar_ref, self.download_dir)
    
    def _load_movie_data(self):
        """Loads the movie metadata into a DataFrame with proper column names."""
        movie_file = self.extracted_dir / "movie.metadata.tsv"
        
        movie_columns = [
            'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_Name', 'Movie_release_Date',
            'Movie_box_office_revenue', 'Movie_runtime', 'Movie_languages', 'Movie_countries', 'Movie_genres'
        ]
        
        if movie_file.exists():
            self.movie_df = pd.read_csv(movie_file, sep='\t', header=None, encoding='utf-8')
            self.movie_df.columns = movie_columns  
            print("Movie data successfully loaded into DataFrame with column names.")
            print(self.movie_df.head())
        else:
            raise FileNotFoundError(f"Movie data file not found at {movie_file}")

    def _load_character_data(self):
        """Loads the character metadata into a DataFrame with proper column names."""
        character_file = self.extracted_dir / "character.metadata.tsv"
        
        character_columns = [
            'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date',
            'Character_name', 'Actor_date_of_birth', 'Actor_Gender', 'Actor_height',
            'Actor_ethnicity', 'Actor_name', 'Actor_age_at_movie_release',
            'Freebase_character/actor_map_ID', 'Freebase_character_ID', 'Freebase_actor_ID'
        ]
        
        if character_file.exists():
            self.character_df = pd.read_csv(character_file, sep='\t', header=None, encoding='utf-8')
            self.character_df.columns = character_columns  
            print("Character data successfully loaded into DataFrame with column names.")
            print(self.character_df.head())
        else:
            raise FileNotFoundError(f"Character data file not found at {character_file}")

            
# Create an instance and call setup() to run everything
movie_data = MovieData()
import ast

def clean_column_values(df: pd.DataFrame, column_name: str):
    """Cleans a column by extracting the genre, country, or language names."""
    if column_name in df.columns:
        # Use ast.literal_eval to safely convert the string to a dictionary
        df[column_name] = df[column_name].apply(
            lambda x: ', '.join(list(ast.literal_eval(x).values())) if isinstance(x, str) else ''
        )
        print(f"{column_name} successfully cleaned.")
    else:
        print(f"Column '{column_name}' not found in DataFrame.")

# Clean the columns for Movie_genres, Movie_countries, and Movie_languages
clean_column_values(movie_data.movie_df, 'Movie_genres')
clean_column_values(movie_data.movie_df, 'Movie_countries')
clean_column_values(movie_data.movie_df, 'Movie_languages')

# Print cleaned data
print(movie_data.movie_df[['Movie_Name', 'Movie_genres', 'Movie_countries', 'Movie_languages']].head())
# Check for null and NaN values in movie_df
print("Null and NaN values in movie_df:")
print(movie_data.movie_df.isnull().sum())

# Check for null and NaN values in character_df
print("Null and NaN values in character_df:")
print(movie_data.character_df.isnull().sum())
# Identify columns with null or NaN values in movie_df
null_columns_movie_df = movie_data.movie_df.columns[movie_data.movie_df.isnull().any()]

# Calculate skewness for columns with null or NaN values in movie_df
print("Skewness of columns with null or NaN values in movie_df:")
for column in null_columns_movie_df:
    if pd.api.types.is_numeric_dtype(movie_data.movie_df[column]):
        skewness = movie_data.movie_df[column].dropna().skew()
        print(f"{column}: {skewness}")
    else:
        print(f"{column}: Non-numeric data, skewness calculation skipped.")

# Identify columns with null or NaN values in character_df
null_columns_character_df = movie_data.character_df.columns[movie_data.character_df.isnull().any()]

# Calculate skewness for columns with null or NaN values in character_df
print("Skewness of columns with null or NaN values in character_df:")
for column in null_columns_character_df:
    if pd.api.types.is_numeric_dtype(movie_data.character_df[column]):
        skewness = movie_data.character_df[column].dropna().skew()
        print(f"{column}: {skewness}")
    else:
        print(f"{column}: Non-numeric data, skewness calculation skipped.")
# Fill NaN values in numeric columns of movie_df with their respective medians
for column in null_columns_movie_df:
    if pd.api.types.is_numeric_dtype(movie_data.movie_df[column]):
        median_value = movie_data.movie_df[column].median()
        movie_data.movie_df[column] = movie_data.movie_df[column].fillna(median_value)
        print(f"Filled NaN values in '{column}' with median value {median_value}")

# Fill NaN values in numeric columns of character_df with their respective medians
for column in null_columns_character_df:
    if pd.api.types.is_numeric_dtype(movie_data.character_df[column]):
        median_value = movie_data.character_df[column].median()
        movie_data.character_df[column] = movie_data.character_df[column].fillna(median_value)
        print(f"Filled NaN values in '{column}' with median value {median_value}")
def movie_type(self, N: int = 10) -> pd.DataFrame:
    """Returns the N most common movie types with their counts."""
    column_with_genres = 'Movie_genres'
    if self.movie_df is None:
        raise ValueError("Data has not been loaded. Run setup() first.")
    
    genre_counts = self.movie_df[column_with_genres].dropna().explode().value_counts().head(N)
    return genre_counts.reset_index().rename(columns={"index": "Genre", 0: "Count"})

def actor_count(self) -> pd.DataFrame:
    """Returns the distribution of the number of actors per movie."""
    column_with_movie_id = 'Wikipedia_movie_ID'  
    if self.character_df is None:
        raise ValueError("Data has not been loaded. Run setup() first.")
    
    actor_distribution = self.character_df.groupby(column_with_movie_id).size()
    return actor_distribution.value_counts().reset_index().rename(columns={"index": "Number of Actors", 0: "Movie Count"})

def actor_distributions(self, gender: str, max_height: float, min_height: float, plot: bool = False) -> pd.DataFrame:
    """Returns actors filtered by gender and height range. Optionally plots the height distribution."""
    column_with_gender = 'Actor_Gender' 
    column_with_height =  'Actor_height' 

    if self.character_df is None:
        raise ValueError("Data has not been loaded. Run setup() first.")

    # Validate inputs
    if gender.lower() not in ["m", "f"]:
        raise ValueError("Gender must be 'm' or 'f'.")
    if not (isinstance(min_height, (int, float)) and isinstance(max_height, (int, float))):
        raise TypeError("Height values must be numeric.")
    if min_height > max_height:
        raise ValueError("min_height must be less than max_height.")

    # Filtering data
    filtered_df = self.character_df[
        (self.character_df[column_with_gender].str.lower() == gender.lower()) &
        (self.character_df[column_with_height].between(min_height, max_height))
    ]

    # Plot if requested
    if plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 5))
        sns.histplot(filtered_df[column_with_height].dropna(), bins=20, kde=True)
        plt.xlabel("Height (cm)")
        plt.ylabel("Count")
        plt.title(f"Height Distribution of {gender.upper()} Actors")
        plt.show()

    return filtered_df

# Attach these methods to the MovieData class
MovieData.movie_type = movie_type
MovieData.actor_count = actor_count
MovieData.actor_distributions = actor_distributions
print(movie_data.movie_type(5))
print(movie_data.actor_count())
print(movie_data.actor_distributions("m", 200, 100, plot=True))
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


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