from elasticsearch import Elasticsearch
from script1 import insert_movies, mainLoop1
from script2 import mainLoop2
es = Elasticsearch(HOST="http://localhost", PORT="9200")

# insert movies
# insert_movies(es)

# ---------------------------search for a movie based only on BM25 metric | Q1---------------------------
print("\n--------------------------------BM25 similarity--------------------------------\n")
mainLoop1(es)
print("\n-------------------------------------------------------------------------------\n")
# -------------------------------------------------------------------------------------------------------

# --------------search for a movie based on BM25, user's rating and avg of all ratings | Q2--------------
print("\n-----------------BM25, user's rating and avg rating similarity-----------------\n")
mainLoop2(es)
print("\n-------------------------------------------------------------------------------\n")
# -------------------------------------------------------------------------------------------------------

