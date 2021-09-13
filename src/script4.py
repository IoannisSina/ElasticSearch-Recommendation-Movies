import os
import string
import warnings
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from gensim.models import word2vec
from keras import losses
from keras.models import Sequential
from keras.layers import Dense, Softmax
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from script1 import search_movies

# All possible ratings. These are our classes for the model output. If the model predocts label 5 then the predicted rating is 3
CLASSES = [.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # avoid tesor flow warnings

def get_ratings_average():
    """
    returns a df with the avg of each movie and the initial df without the timestamp
    """
    ratings_avg = ratings.groupby(by='movieId').mean()
    ratings_avg = ratings_avg.drop('userId', axis=1).reset_index()

    return ratings_avg

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

def getPredictionLabels(predictions):
    """
    This function calculates the real label for each prediction.
    Returns the index of the biggest probability
    """
    return [np.argmax(prediction) for prediction in predictions]

def neuralForUser(user_id):
    """
    This function creates and trains a neural network according to the current user's
    ratings. Return trained model in order to use it for predicting movies' ratings.
    Return movie vectors so we dont have to recalculate them.
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
            y.append(LABELS[CLASSES.index(np.array(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating']))]) # append the LABEL of the class

    # split X, y in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=42)

    # define the keras model
    model = Sequential()
    model.add(Dense(32, input_dim=len(X_train[0]), activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10))

    # compile the keras model
    model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

    print("\n------------------------------Training model------------------------------\n")
    # fit the keras model on the dataset
    model.fit(np.array(X_train), np.array(y_train), epochs=10, verbose=0)
    print("\n--------------------------------------------------------------------------\n")

    # make predictions of the test set. Each prediction will be an array which will provide a probability 
    # for each label. We will keep the biggest one
    probability_model = Sequential([model, Softmax()])

    # see accurancy and 
    predictions = probability_model.predict(np.array(X_test))

    # each prediction is a 10 length vector. We have to calculate y_pred based on the boggest probability
    y_pred = getPredictionLabels(predictions)

    # assert y_test and y_pred have same length
    assert len(y_test) == len(y_pred), "Something is wrong with the predicsion process!"

    print("\n-------------------------------Scores-------------------------------\n")
    print("Precision: " + str(precision_score(y_test, y_pred, labels=LABELS, average='micro')))
    print("Recall: " + str(recall_score(y_test, y_pred, labels=LABELS, average='micro')))
    print("F1 Score: " + str(f1_score(y_test, y_pred, labels=LABELS, average='micro')))
    print("\n--------------------------------------------------------------------\n")
    return probability_model, movie_vectors

def final_rating(response, user_id, predictor, movies_vectors):
    # get all dfs needed to cacluate the final one and create final result
    final_result = []
    ratings_avg = get_ratings_average()
    # I will use a linear combination of BM25 user's rating and avg_ratings for the new metric
    # change all scores according to the metric: BM25 + user's rating after trainings the model for the current user + avg rating
    for movie in response:
        movie_id = movie['_source']['movieId']
        # if avg does not exist set it to 0 so it does not affect the sum
        # if user's rating does not exist set it to 0 so it does not affect the sum
        movie_BM25 = movie['_score']
        movie_ratings_avg = float(ratings_avg.loc[ratings_avg['movieId'] == movie_id].iloc[0]['rating']) if movie_id in ratings_avg.movieId else -1
        user_rating = float(ratings.loc[(ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)].iloc[0]['rating']) if ((ratings['movieId'] == movie_id) & (ratings['userId'] == user_id)).any() else "-"
        # assert that row exsists!!
        # -----------------------------------------calculate predicted RATING-----------------------------------------
        prediction_array = predictor.predict(np.array([movies_vectors[movie_id]]))
        predicted_label = getPredictionLabels(prediction_array)
        user_predicted_rating = CLASSES[predicted_label[-1]]
        # ------------------------------------------------------------------------------------------------------------
        new_record = {
            "Title": movie['_source']['title'],
            "Score": movie_BM25
        }
        new_record["Score"] += movie_ratings_avg if movie_ratings_avg != -1 else 0
        new_record["Score"] += user_rating if user_rating != "-" else user_predicted_rating
        new_record["User_true_r"] = user_rating
        new_record["User_predicted_r"] = user_predicted_rating
        new_record["Genres"] = movie['_source']['genres']
        final_result.append(new_record)
    
    df = pd.DataFrame(final_result)
    df.sort_values("Score", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    return df

def mainLoop4(es):
    warnings.simplefilter("ignore")
    user_id = input("Please enter you ID (int): \n")
    while (not user_id.isdigit()):
        user_id = input("Please enter user's ID (int): \n")
    
    # calculate here in order to do it once
    predictor, movies_vectors = neuralForUser(int(user_id))
    title_input = input("Please enter a movie title (type exit() to exit):\n")
    while(title_input != "exit()"):
        response = search_movies(es, title_input)
        if len(response['hits']['hits']) == 0:
            print("\nNo movies returned!\n")
        else:
            print("\n\nMovies similar to " + title_input + " (BM25, avarage ratings and user's rating AFTER NN predictions):\n")
            print(final_rating(response['hits']['hits'], int(user_id), predictor, movies_vectors))
            print("\n")
        title_input = input("Please enter a movie title (type exit() to exit):\n")
    print("\n")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False)
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    genres = get_all_genres() # read genres here so we have the same order though the whole execution
    mainLoop4(es)