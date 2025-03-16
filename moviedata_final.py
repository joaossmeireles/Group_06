import os
import tarfile
import requests
import pandas as pd
from pathlib import Path
import ast
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

class MovieData:
    data_url = os.getenv(
        "DATA_URL", "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz")
    download_dir = Path("downloads")
    extracted_dir = download_dir / "MovieSummaries"
    movie_df = None
    character_df = None

    def __init__(self):
        """Runs the download, extraction, and data loading process."""
        self.download_dir.mkdir(exist_ok=True)
        self._download_data()
        self._extract_data()
        self._load_movie_data()
        self._load_character_data()
        self._clean_data()

    def _download_data(self):
        """Downloads the dataset if it doesn't already exist."""
        file_path = self.download_dir / "MovieSummaries.tar.gz"
        if not file_path.exists():
            logging.info("Downloading dataset...")
            try:
                response = requests.get(self.data_url, stream=True, timeout=30)
                response.raise_for_status()

                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                logging.info("Download complete.")
            except Exception as e:
                logging.error(f"Failed to download data: {e}")
                raise
        else:
            logging.info("Dataset already exists. Skipping download.")

    def _extract_data(self):
        """Extracts the dataset if not already extracted."""
        file_path = self.download_dir / "MovieSummaries.tar.gz"
        if not self.extracted_dir.exists():
            logging.info("Extracting dataset...")
            try:
                with tarfile.open(file_path, "r:gz") as tar_ref:
                    tar_ref.extractall(path=self.download_dir)
                logging.info("Extraction complete.")
            except Exception as e:
                logging.error(f"Failed to extract data: {e}")
                raise
        else:
            logging.info("Dataset already extracted. Skipping extraction.")

    def _load_movie_data(self):
        """Loads movie data into a pandas DataFrame."""
        movie_file = self.extracted_dir / "movie.metadata.tsv"
        movie_columns = [
            'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_Name', 'Movie_release_Date',
            'Movie_box_office_revenue', 'Movie_runtime', 'Movie_languages', 'Movie_countries', 'Movie_genres'
        ]

        if movie_file.exists():
            self.movie_df = pd.read_csv(movie_file, sep='\t', header=None, encoding='utf-8', low_memory=False)
            self.movie_df.columns = movie_columns
            logging.info("Movie data successfully loaded.")
        else:
            raise FileNotFoundError(f"Movie data file not found at {movie_file}")

    def _load_character_data(self):
        """Loads character data into a pandas DataFrame."""
        character_file = self.extracted_dir / "character.metadata.tsv"

        # 🔍 Check if file exists
        if not character_file.exists():
            logging.error(f"Character data file not found at {character_file}")
            raise FileNotFoundError(f"Character data file not found at {character_file}")
        
        # Print file size for debugging
        logging.info(f"Character file found: {character_file}, Size: {character_file.stat().st_size} bytes")

        # Read file
        try:
            self.character_df = pd.read_csv(character_file, sep='\t', header=None, encoding='utf-8', low_memory=False)
            logging.info("Character data successfully loaded.")

            # Debug output: Show first few rows
            print("DEBUG: First 5 rows of character_df:\n", self.character_df.head())

        except Exception as e:
            logging.error(f"Failed to read character data: {e}")
            raise

        character_columns = [
            'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_release_date',
            'Character_name', 'Actor_date_of_birth', 'Actor_Gender', 'Actor_height',
            'Actor_ethnicity', 'Actor_name', 'Actor_age_at_movie_release',
            'Freebase_character/actor_map_ID', 'Freebase_character_ID', 'Freebase_actor_ID'
        ]

        if character_file.exists():
            self.character_df = pd.read_csv(character_file, sep='\t', header=None, encoding='utf-8', low_memory=False)
            self.character_df.columns = character_columns
            logging.info("Character data successfully loaded.")
        else:
            raise FileNotFoundError(f"Character data file not found at {character_file}")
    def _clean_data(self):
        """Cleans up genre, country, and language columns."""
        def clean_column_values(df, column):
            if column in df.columns:
                def safe_eval(x):
                    try:
                        if isinstance(x, str):
                            return ', '.join(list(ast.literal_eval(x).values()))
                    except Exception as e:
                        logging.warning(f'Failed to parse {column}: {x} - {e}')
                        return ''
                df[column] = df[column].apply(safe_eval)

        clean_column_values(self.movie_df, 'Movie_genres')
        clean_column_values(self.movie_df, 'Movie_countries')
        clean_column_values(self.movie_df, 'Movie_languages')

        # Clean up height data and standardize gender values
        self.character_df['Actor_height'] = pd.to_numeric(self.character_df['Actor_height'], errors='coerce')
        self.character_df.loc[self.character_df["Actor_height"] < 10, "Actor_height"] *= 100
        print("DEBUG: Heights After Conversion to CM:")
        print(self.character_df["Actor_height"].unique())

        self.character_df['Actor_Gender'] = (
            self.character_df['Actor_Gender']
            .fillna("unknown")
            .str.lower().str.strip()
            .map({"m": "male", "f": "female"})
            .fillna(self.character_df['Actor_Gender'])
        )
        self.character_df.dropna(subset=['Actor_height'], inplace=True)

    def movie_type(self, N: int = 10) -> pd.DataFrame:
        """Returns the N most common movie types with their counts."""
        if not isinstance(N, int):
            raise ValueError("N must be an integer.")
    
        if self.movie_df is not None and 'Movie_genres' in self.movie_df.columns:
            genre_counts = (self.movie_df['Movie_genres']
                            .dropna()
                            .str.split(', ')
                            .explode()
                            .str.strip()
                            .value_counts()
                            .head(N)
                            .reset_index())

            genre_counts.columns = ['Movie_Type', 'Count']
            return genre_counts
        else:
            return pd.DataFrame()

    def releases(self, genre=None):
        """Returns a DataFrame counting movies released per year, optionally filtering by genre."""
        if self.movie_df is None or 'Movie_release_Date' not in self.movie_df.columns:
            raise ValueError("Movie data not loaded correctly.")
        
        df = self.movie_df.copy()
        df['Year'] = pd.to_datetime(df['Movie_release_Date'], errors='coerce').dt.year
        df = df.dropna(subset=['Year'])
        
        if genre:
            df = df[df['Movie_genres'].str.contains(genre, na=False, case=False)]
        
        return df.groupby('Year').size().reset_index(name='Movie_Count')

    def actor_count(self) -> pd.DataFrame:
        """Returns the distribution of the number of actors per movie."""
        print("DEBUG: Checking if character_df is loaded...")
        if self.character_df is None or self.character_df.empty:
            logging.error("character_df is None or empty. Data not loaded")
            return pd.DataFrame()
        print("DEBUG: First 5 rows of character_df before actor count:")
        print(self.character_df.head(5))

        actor_count_df = (
            self.character_df.groupby("Wikipedia_movie_ID")  # Count actors per movie
            .size()
            .reset_index(name="Number of Actors"))  # Rename column
        actor_count_df = (
            actor_count_df.groupby("Number of Actors")
            .size()
            .reset_index(name="Movie Count"))  # Rename for clarity
        print("DEBUG: First 5 rows of actor_count_df:")
        print(actor_count_df.head(5))
        
        return actor_count_df
    
    def ages(self, unit='Y'):
        """Returns a DataFrame counting actor births per year ('Y') or month ('M')."""
        if self.character_df is None or 'Actor_date_of_birth' not in self.character_df.columns:
            raise ValueError("Character data not loaded correctly.")
        
        df = self.character_df.copy()
        df['Date'] = pd.to_datetime(df['Actor_date_of_birth'], errors='coerce')
        
        if unit == 'M':
            df['Month'] = df['Date'].dt.month
            return df.groupby('Month').size().reset_index(name='Birth_Count')
        else:
            df['Year'] = df['Date'].dt.year
            return df.groupby('Year').size().reset_index(name='Birth_Count')

    def actor_distributions(self, gender="all", min_height=150, max_height=200, plot=False) -> pd.DataFrame:
        """Returns actors filtered by gender and height range."""
        if self.character_df is None or self.character_df.empty:
            logging.error("character_df is None or empty. Data not loaded.")
            return pd.DataFrame()

        self.character_df['Actor_Gender'] = (
            self.character_df['Actor_Gender']
            .fillna("unknown")
            .str.lower().str.strip()
            .map({"m": "male", "f": "female"})
            .fillna(self.character_df['Actor_Gender'])
        )

        print("DEBUG: Unique Heights Before Filtering:", self.character_df["Actor_height"].unique())
        filtered_df = self.character_df[
            (self.character_df['Actor_height'] >= min_height) &
            (self.character_df['Actor_height'] <= max_height)
        ]
        print("DEBUG: Unique Heights After Filtering:", filtered_df["Actor_height"].unique())

        if gender.lower() != "all":
            filtered_df = filtered_df[filtered_df['Actor_Gender'].str.lower() == gender.lower()]

        if filtered_df.empty:
            logging.warning(f"No data found for gender={gender}, height range=({min_height}, {max_height})")
            return pd.DataFrame()
        
        return filtered_df[['Actor_Gender', 'Actor_height']]

# Initialize class for testing
if __name__ == '__main__':
    data = MovieData()
    print(data.movie_type(10))
    print(data.actor_count())
    print(data.actor_distributions(gender="male", min_height=150, max_height=200, plot=False))

