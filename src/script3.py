import os
import warnings
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from script1 import search_movies

def get_ratings_average():
    """
    returns a df with the avg of each movie and the initial df without the timestamp
    """
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    ratings = ratings[['userId','movieId','rating']]

    ratings_avg = ratings.groupby(by='movieId').mean()
    ratings_avg = ratings_avg.drop('userId', axis=1).reset_index()

    return ratings_avg

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

def kMeans():
    X = allUsersGenreRatings()
    kmeans = KMeans(
        init="random",
        n_clusters=3,
        n_init=10,
        max_iter=1000,
        random_state=42
    )

    predictions = kmeans.fit_predict(X)
    return kmeans.labels_

def calculateRating(clustered_users, movie_id, user_id):
    ratings_sum = 0.0
    ratings_count = 0.0

    for user_id in clustered_users:
        # if user has rating
        if ((ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)).any():
            ratings_sum += float(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating'])
            ratings_count += 1
    return round((ratings_sum / ratings_count), 2) if ratings_count != 0 else 0

def fillAllRatings(movies):
    """
    call this ONLY for returned movies. We dont have to process all movies.
    Fill all non existing ratings using predictions. I will check if rating already exists.
    If not take all users of the cluster and take the mean of their ratings for the movie
    """
    labels = kMeans()
    new_ratings = []
    print("Final result must be num of movies * num of users = " + str(len(movies)) + " * " + str(len(labels)) + " = " + str(len(movies) * len(labels)) + " rows \n")

    # labels' length equals num of users!! index i represents i + 1 id
    # add a new column which indicates if rating copmes from clustering or already exists
    for user_id in range(1, len(labels) + 1):
        # get all users in the same cluster and pass them to calculate ratings
        other_users_in_cluster = [i + 1 for i, x in enumerate(labels) if x == user_id]

        for movie in movies:
            movie_id = movie['_source']['movieId']
            if ((ratings['movieId'] == movie_id) & (ratings['userId'] == (user_id))).any():
                # rating already exists so push it to new_ratings
                new_rating = {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": float(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating']),
                    "From_cluster": "False"
                }
            else:
                # if rating does not exist, calculate it from the clusters
                new_rating = {
                    "userId": user_id,
                    "movieId": movie_id,
                    "rating": calculateRating(other_users_in_cluster, movie_id, user_id),
                    "From_cluster": "True"
                }
            new_ratings.append(new_rating)
    
    df = pd.DataFrame(new_ratings)
    assert len(df.index) == len(movies) * len(labels), "Something is wrong"
    return df

def final_rating(movies, user_id):
    # get all dfs needed to cacluate the final one and create final result
    final_result = []
    ratings_avg = get_ratings_average()
    cluster_ratings = fillAllRatings(movies) # df with all ratings existing (for returned movies)
    # I will use a linear combination of BM25 user's rating and avg_ratings for the new metric
    # change all scores according to the metric: BM25 + user's rating after clustering on the movie + avg rating
    for movie in movies:
        movie_id = movie['_source']['movieId']
        # if avg does not exist set it to 0 so it does not affect the sum
        # if user's rating does not exist set it to 0 so it does not affect the sum
        movie_BM25 = movie['_score']
        movie_ratings_avg = float(ratings_avg.loc[ratings_avg['movieId'] == movie_id].iloc[0]['rating']) if movie_id in ratings_avg.movieId else -1
        # assert that row exsists!!
        assert ((cluster_ratings['movieId'] == movie_id) & (cluster_ratings['userId'] == user_id)).any(), "Rating does not exist! K means is wrong!"
        cluster_rating = float(cluster_ratings.loc[(cluster_ratings['movieId'] == movie_id) & (cluster_ratings['userId'] == user_id)].iloc[0]['rating'])

        new_record = {
            "Title": movie['_source']['title'],
            "Score": movie_BM25 + cluster_rating
        }
        new_record["Score"] += movie_ratings_avg if movie_ratings_avg != -1 else 0
        new_record["Average_r"] = movie_ratings_avg if movie_ratings_avg != -1 else "-"
        new_record["User_cluster_r"] = cluster_rating
        new_record["Genres"] = movie['_source']['genres']
        final_result.append(new_record)
    
    df = pd.DataFrame(final_result)
    df.sort_values("Score", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    return df

def mainLoop3(es):
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
            print("\n\nMovies similar to " + title_input + " (BM25, avarage ratings and user's rating AFTER K-means):\n")
            print(final_rating(response['hits']['hits'], int(user_id)))
            print("\n")
        title_input = input("Please enter a movie title (type exit() to exit):\n")
    print("\n")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    mainLoop3(es)