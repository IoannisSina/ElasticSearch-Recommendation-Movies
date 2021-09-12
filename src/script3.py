import os
import warnings
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
import pandas as pd

def get_all_genres():
    """
    Return all unique genres in order to calculate mean for users
    """
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False)
    pre_genres = movies["genres"].to_list()
    genres = []

    # get all unique genres
    for string in pre_genres:
        current_genres = string.split("|")

        for genre in current_genres:
            if genre not in genres:
                genres.append(genre)

    return genres


def allUsersGenreRatings():
    """
    Calculate for each user the mean of his/her ratings for all genres
    671 users and 20 genres. The returnd df will be 671 x 20
    """
    warnings.filterwarnings("ignore", 'This pattern has match groups')
    genre_ratings = pd.DataFrame()
    genres = get_all_genres()

    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False)

    for genre in genres:
        genre_movies = movies[movies['genres'].str.contains(genre)]
        avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, ['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
        
        genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
    
    genre_ratings = genre_ratings.fillna(0)
    genre_ratings.columns = genres
    return genre_ratings

def mainLoop3(es):
    # print(allUsersGenreRatings())
    print("Test")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    mainLoop3(es)