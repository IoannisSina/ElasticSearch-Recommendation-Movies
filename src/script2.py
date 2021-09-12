import os
import warnings
from elasticsearch import Elasticsearch
import pandas as pd
# import numpy as np

from script1 import search_movies

def get_ratings_average():
    """
    returns a df with the avg of each movie and the initial df without the timestamp
    """
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    ratings = ratings[['userId','movieId','rating']]

    ratings_avg = ratings.groupby(by='movieId').mean()
    ratings_avg = ratings_avg.drop('userId', axis=1).reset_index()

    return ratings, ratings_avg

# sample of returned value
# {'_index': 'movies', '_type': '_doc', '_id': '1', '_score': 8.264357, '_source': {'movieId': 1, 'title': 'Toy Story (1995)', 'genres': 'Adventure|Animation|Children|Comedy|Fantasy'}}
def final_rating(response, user_id):
    # get all dfs needed to cacluate the final one and create final result
    final_result = []
    ratings, ratings_avg = get_ratings_average()

    # I will use a linear combination of BM25 user's rating and avg_ratings for the new metric
    # change all scores according to the metric: BM25 + user's rating on the movie + avg rating
    for movie in response:
        movie_id = movie['_source']['movieId']

        # if avg does not exist set it to 0 so it does not affect the sum
        # if user's rating does not exist set it to 0 so it does not affect the sum
        movie_BM25 = movie['_score']
        movie_ratings_avg = float(ratings_avg.loc[ratings_avg['movieId'] == movie_id].iloc[0]['rating']) if movie_id in ratings_avg.movieId else -1
        movie_rating_user = float(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating']) if ((ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)).any() else -1

        new_record = {
            "Title": movie['_source']['title'],
            "Score": movie_BM25
        }
        new_record["Score"] += movie_ratings_avg if movie_ratings_avg != -1 else 0
        new_record["Score"] += movie_rating_user if movie_rating_user != -1 else 0
        new_record["Average_r"] = movie_ratings_avg if movie_ratings_avg != -1 else "-"
        new_record["User_r"] = movie_rating_user if movie_rating_user != -1 else "-"
        new_record["Genres"] = movie['_source']['genres']
        final_result.append(new_record)
    
    df = pd.DataFrame(final_result)
    df.sort_values("Score", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    return df

def mainLoop2(es):
    warnings.simplefilter("ignore")
    user_id = input("Please enter you ID (int): \n")
    while (not user_id.isdigit()):
        user_id = input("Please enter user's ID (int): \n")
    
    title_input = input("Please enter a movie title (type exit() to exit):\n")
    while(title_input != "exit()"):
        response = search_movies(es, title_input)
        if len(response['hits']['hits']) == 0:
            print("\nNo movies returned!\n")
        else:
            print("\n\nMovies similar to " + title_input + " (BM25, avarage ratings and user's rating):\n")
            print(final_rating(response['hits']['hits'], int(user_id)))
            print("\n")
        title_input = input("Please enter a movie title (type exit() to exit):\n")
    print("\n")


if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    mainLoop2(es)