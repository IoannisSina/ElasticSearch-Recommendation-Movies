from elasticsearch import Elasticsearch
from script1 import insert_movies, mainLoop1
from script2 import mainLoop2
from script3 import mainLoop3
from script4 import mainLoop4
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

# ------search for a movie based on BM25, user's rating and avg of all ratings (after K-means) | Q3------
print("\n------------BM25, user's rating and avg rating similarity (K-means)------------\n")
mainLoop3(es)
print("\n-------------------------------------------------------------------------------\n")
# -------------------------------------------------------------------------------------------------------

# --search for a movie based on BM25, user's rating and avg of all ratings (after Neural network) | Q4---
print("\n---------BM25, user's rating and avg rating similarity (Neural Network)--------\n")
mainLoop4(es)
print("\n-------------------------------------------------------------------------------\n")
# -------------------------------------------------------------------------------------------------------
