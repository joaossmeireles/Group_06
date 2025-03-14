import os
import tarfile
import urllib.request
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns

class MovieData:
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'MovieSummaries'))
    data_url = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    archive_name = "MovieSummaries.tar.gz"
    movie_df = "movie.metadata.tsv"
    character_df = "character.metadata.tsv"

    genre_mapping = {
        "/m/07s9rl0": "Drama",
        "/m/01z4y": "Comedy",
        "/m/02l7c8": "Action",
        "/m/01g6gs": "Romance",
        "/m/02kdv5l": "Horror",
        "/m/01jfsb": "Thriller",
        "/m/02hmvc": "Science Fiction",
        "/m/03q4nz": "Adventure",
        "/m/0lsxr": "Mystery",
        "/m/0219x_": "Animation",
        "/m/02vxn": "Crime",
        "/m/09b5t": "Fantasy",
        "/m/06ntj": "Family",
        "/m/018jz": "Musical",
        "/m/07c6l": "Biography",
        "/m/01h6rj": "War",
        "/m/03tmr": "History",
        "/m/06bm2": "Western",
        "/m/07v9_z": "Sport",
        "/m/0f2f9": "Music",
        "/m/019_rr": "Documentary",
        "/m/0jtdp": "Political Cinema",
        "/m/03npn": "Rockumentary",
        "/m/06ppq": "Indie",
        "/m/03k9fj": "Romantic Comedy",
        "/m/0hqxf": "Romantic Drama",
        "/m/03btsm8": "World Cinema",
        "/m/05p553": "Psychological Thriller",
        "/m/04t36": "Musical",
        "/m/0hcr": "Gay Themed",
        "/m/068d7h": "Crime Thriller"
    }

    def __init__(self):
        os.makedirs(self.download_dir, exist_ok=True)

        if not self._check_data_files():
            self._download_and_extract_data()

        self.movies_df = self._load_movies()
        self.characters_df = self._load_characters()
        self.merged_df = self._merge_data()

    def _check_data_files(self):
        """Verify whether the necessary data files are present in the downloads directory."""
        return (
            os.path.exists(os.path.join(self.download_dir, self.movie_df)) and
            os.path.exists(os.path.join(self.download_dir, self.character_df))
        )

    def _download_and_extract_data(self):
        """Download and extract the dataset if it is not found."""
        archive_path = os.path.join(self.download_dir, self.archive_name)

        if not os.path.exists(archive_path):
            print("Downloading dataset...")
            urllib.request.urlretrieve(self.data_url, archive_path)
            print("Download complete.")

        print("Extracting files...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.download_dir)
        print("Extraction complete.")
                  
    def _load_movies(self):
        """Load movies dataset"""
        file_path = os.path.join(self.download_dir, self.movie_df)
        columns = [
            "wikipedia_movie_id", "freebase_movie_id", "movie_name",
            "release_date", "box_office", "runtime", "languages",
            "countries", "genres"
        ]
        df = pd.read_csv(file_path, sep="\t", header=None, names=columns)

        def extract_genre_ids(genre_dict_str):
            try:
                genre_dict = ast.literal_eval(genre_dict_str)
                return [self.genre_mapping.get(gid, f"Unknown ({gid})") for gid in genre_dict.keys()]
            except (ValueError, SyntaxError):
                return []

        df["genres"] = df["genres"].apply(lambda x: extract_genre_ids(x) if isinstance(x, str) else [])
        return df

    def _load_characters(self):
        """Load characters dataset"""
        file_path = os.path.join(self.download_dir, self.character_df)
        columns = [
            "wikipedia_movie_id", "freebase_movie_id", "release_date",
            "character_name", "actor_dob", "actor_gender", "actor_height",
            "actor_ethnicity", "actor_name", "actor_age_at_release",
            "character_actor_map_id", "character_id", "actor_id"
        ]
        return pd.read_csv(file_path, sep="\t", header=None, names=columns)

    def _merge_data(self):
        """Merge movies and characters dataframes"""
        return pd.merge(self.characters_df, self.movies_df, on="wikipedia_movie_id", how="inner")

    def clean_data(self):
        """Clean the merged data"""
        self.merged_df["actor_gender"] = self.merged_df["actor_gender"].replace({"M": "Male", "F": "Female"})

    def movie_type(self, n=10):
        """Return the top N movie types by count"""
        genre_counts = pd.Series(
           [genre for sublist in self.movies_df["genres"] for genre in sublist]
        ).value_counts()
        return genre_counts.head(n).reset_index().rename(columns={"index": "Genre", 0: "Count"})

    def actor_count(self):
        return self.characters_df['actor_id'].value_counts().reset_index().rename(columns={'index': 'Actor ID', 'actor_id': 'Count'})

    def actor_distribution(self, gender=None, min_height=150, max_height=200, plot=False):
        filtered_df = self.characters_df[
            (self.characters_df['actor_height'] >= min_height) &
            (self.characters_df['actor_height'] <= max_height)
        ]

        if gender:
            filtered_df = filtered_df[filtered_df['actor_gender'].str.lower() == gender.lower()]

        if plot:
            plt.figure(figsize=(8, 5))
            sns.histplot(filtered_df['actor_height'], bins=20, kde=True)
            plt.xlabel("Height (cm)")
            plt.ylabel("Count")
            plt.title(f"Actor Height Distribution ({gender.capitalize() if gender else 'All'})")
            plt.show()

        return filtered_df[['actor_gender', 'actor_height']]

    def __repr__(self):
        return f"<MovieData loaded: {len(self.movies_df)} movies, {len(self.characters_df)} characters>"

if __name__ == "__main__":
    data = MovieData()
    print(data.movie_type())
    print(data.actor_count())
