import os
import warnings
from elasticsearch import Elasticsearch
import matplotlib.pyplot as plt
from gensim.models import word2vec
import pandas as pd
from script1 import search_movies

def titlesToVec():
    movies_titles = movies["title"].to_list()
    tokenized_titles = [title.split() for title in movies_titles]
    print(len(tokenized_titles))
    model = word2vec.Word2Vec(tokenized_titles, min_count=1)
    print(len(model.wv))

def mainLoop4(es):
    titlesToVec()
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
    #         print(final_rating(response['hits']['hits'], int(user_id)))
    #         print("\n")
    #     title_input = input("Please enter a movie title (type exit() to exit):\n")
    # print("\n")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://localhost", PORT="9200")
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False)
    ratings = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\\ratings.csv", index_col=False)
    mainLoop4(es)