import os
import string
import warnings
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from gensim.models import word2vec
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from script1 import search_movies

# All possible ratings. These are our classes for the model output. If the model predocts label 5 then the predicted rating is 3
CLASSES = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def get_all_genres():
    """
    Return all unique genres in order to calculate mean for users
    """
    pre_genres = movies["genres"].to_list()

    # get all unique genres
    tokenized_genres = [txt.split("|") for txt in pre_genres]
    genres = set([item for sublist in tokenized_genres for item in sublist])

    return list(genres)

def generateTitleVectors():
    """
    Represent the title's of the movies as one vector using word2vec. We take the mean of all word vectors
    Return a dict -> movieId: title vector
    """
    title_vectors = {}

    # here we get a vector for every word in all titles
    movies_titles = movies["title"].to_list()
    movies_ids = movies["movieId"].to_list()
    tokenized_titles = [title.translate(str.maketrans('', '', string.punctuation)).split() for title in movies_titles]
    model = word2vec.Word2Vec(tokenized_titles, min_count=1)
    word_vectors = model.wv

    # Now for each title we get the SUM of the words' vectors. E.g if title is Toy Story the the movie's title vector is
    # word_vectos["Toy"] + word_vector["Story"]
    for movie_id in movies_ids:

        # assert that movie exists
        assert (movies['movieId'] == movie_id).any(), "Movie does not exist!"

        # get title splitted and removed punctuation
        title_words = movies.loc[movies['movieId'] == movie_id].iloc[0]['title'].translate(str.maketrans('', '', string.punctuation)).split(" ")

        temp_vector_1 = np.array(word_vectors.get_vector(title_words[0]))
        for word in title_words[1:]:
            if word != "":
                # print(word_vectors.get_vector(word))
                temp_vector_2 = np.array(word_vectors.get_vector(word))
                temp_vector_1 += temp_vector_2
        title_vectors[movie_id] = (temp_vector_1 / len(title_words))

    assert len(title_vectors) == len(movies_ids), "Something is wrong with the movie titles!"
    return title_vectors

def generateGenreVectors():
    """
    Represent genres as vectors zer-one (one hot encoding) for each movie
    The returned result will be a matrix movies' length * genres' length.
    Return a dict moviId -> genres dict -> [1 0 0 ... 0] 1 where genre belongs to movie else 0
    """
    # genres from main
    genre_vectors = {}
    movies_ids = movies["movieId"].to_list()

    for movie_id in movies_ids:

        # assert that movie exists
        assert (movies['movieId'] == movie_id).any(), "Movie does not exist!"

        # get title splitted and removed punctuation
        movies_genres = movies.loc[movies['movieId'] == movie_id].iloc[0]['genres'].split("|")

        # if genre in movies genres append 1 else 0
        movie_genre_vector = []
        for genre in genres:
            movie_genre_vector.append(1 if (genre in movies_genres) else 0)
        
        assert movie_genre_vector.count(1) == len(movies_genres), "Something is wrong with genres!"
        
        genre_vectors[movie_id] = movie_genre_vector
    
    assert len(genre_vectors) == len(movies_ids), "Something is wrong with the movie genres!"
    return genre_vectors

def generateMovieVectors():
    """
    This function generates a vector for each movie combining the title vectors and the genre vectors.
    For each movie we concatenate the genre vector to the title vector of the movie.
    Final vectors will be title vectors' length + genres' length.
    Return a dict movieId: movie vector
    """
    movie_vectors = {}
    movies_ids = movies["movieId"].to_list()

    # get results from previous versions
    title_vectors = generateTitleVectors()
    genre_vectors = generateGenreVectors()

    # assert lists have same length
    assert len(title_vectors) == len(genre_vectors), "Something is wrong with the final movie vectors"

    for movie_id in movies_ids:
        movie_vectors[movie_id] = list(title_vectors[movie_id]) + list(genre_vectors[movie_id])
    return movie_vectors

def neuralForUser(user_id):
    """
    This function creates and trains a neural network according to the current user's
    ratings. Return trained model in order to use it for predicting movies' ratings
    """
    # for one user
    X = []
    y = []
    movie_vectors = generateMovieVectors()
    movies_ids = movies["movieId"].to_list()

    # create train test set X,Y in order to train our model. X Y are only ucurent user's ratings
    for movie_id in movies_ids:
        if ((ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)).any():
            X.append(movie_vectors[movie_id])
            y.append(float(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating']))
    

    kf = KFold(n_splits=4, shuffle=True)

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]

        # define the keras model
        model = Sequential()
        model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(10, activation='softmax'))
        # model.summary()

        # compile the keras model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit the keras model on the dataset
        model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)




def mainLoop4(es):
    neuralForUser(1)
    # neuralForUser("", 1)
    # warnings.simplefilter("ignore")
    # user_id = input("Please enter you ID (int): \n")
    # while (not user_id.isdigit()):
    #     user_id = input("Please enter user's ID (int): \n")
    
    # title_input = input("Please enter a movie title (type exit() to exit):\n")
    # while(title_input != "exit()"):
    #     response = search_movies(es, title_input)
    #     if len(response['hits']['hits']) == 0:
    #         print("\nNo movies returned!\n")
    #     else:
    #         print("\n\nMovies similar to " + title_input + " (BM25, avarage ratings and user's rating AFTER K-means):\n")
    #         neuralForUsers(title_input, int(user_id))
    #         print("\n")
    #     title_input = input("Please enter a movie title (type exit() to exit):\n")
    # print("\n")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False)
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    genres = get_all_genres() # read genres here so we have the same order though the whole execution
    mainLoop4(es)