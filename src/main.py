from script1 import insert_movies, search_movies
from elasticsearch import Elasticsearch

es = Elasticsearch(HOST="http://localhost", PORT="9200")

# insert movies
# insert_movies(es)

# search for a movie
title_input = "Lady" # input("Please enter a movie title:")
response = search_movies(es, title_input)
print("\n\nMovies similar to " + title_input + " (descending order, BM25 metric):\n")
for movie in response['hits']['hits']:
  print(movie['_source']['title'] + " with a similarity score of " + str(movie['_score']))

