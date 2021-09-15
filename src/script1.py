"""
Read movies.csv and add the to Elasticsearch DB
"""
import os
import warnings
from elasticsearch import Elasticsearch, helpers
import pandas as pd


QUERY_SIZE = 20

def bulk_json_data(movies, _index):
    """
    Returns the record that should be saved to ElasticSeacr Database
    """
    for rec in movies:
        yield {
            "_index": _index,
            "_id": rec["movieId"],
            "_source": rec,
        }

def insert_movies(es):
    """
    Index all movies with one bulk json file
    """
    #Creating index, ignore if already exists
    es.indices.create(index='movies', ignore=400,body={
        "settings":{
            "index":{
                "similarity":{
                    "default":{
                        "type":"BM25"
                    }
                }
            }
        },
        "mappings":{
            "standard_mapping":
            {
                "title":{"analyzer":"english"}
                }
            }
        }
    )
    movies = pd.read_csv(os.path.abspath(os.getcwd()) + "\src\datasets\movies.csv", index_col=False).to_dict('records') # movies as records

    try:
        # bulk call for indexing all movies
        response = helpers.bulk(es,bulk_json_data(movies, 'movies'))
        assert response[0] == len(movies), "Error: Not all movies were inserted succesfully."
        # print(response)
    except Exception as e:
        print("Error: ", e)
    # es.indices.delete(index='movies')

def search_movies(es, title):
    """
    return all movies matching with the given string
    """

    query = {
        # get only first 10 records
        "size": QUERY_SIZE,
        "query": {
            "match": {
                "title": title
            }
        },
        "sort": [
            {"_score": {"order": "desc"}}
        ]
    }

    result = es.search(body=query, index='movies')
    return result

# sample of returned value
# {'_index': 'movies', '_type': '_doc', '_id': '1', '_score': 8.264357, '_source': {'movieId': 1, 'title': 'Toy Story (1995)', 'genres': 'Adventure|Animation|Children|Comedy|Fantasy'}}
def result_to_pd(result):
    """
    Converts the final result to DataFrame in order to print it
    """
    final_result = []
    for movie in result:
        temp = {
            "Title": movie['_source']['title'],
            "Score": movie['_score'],
            "Genres": movie['_source']['genres']
        }
        final_result.append(temp)
    
    df = pd.DataFrame(final_result)
    df.sort_values("Score", inplace=True, ascending=False)
    df = df.reset_index(drop=True)
    return df

def mainLoop1(es):
    warnings.simplefilter("ignore")
    title_input = input("Please enter a movie title (type exit() to exit):\n")
    while(title_input != "exit()"):
        response = search_movies(es, title_input)

        if len(response['hits']['hits']) == 0:
            print("\nNo movies returned!\n")
        else:
            print("\n\nMovies similar to " + title_input + " (descending order, BM25 metric):\n")
            print(result_to_pd(response['hits']['hits']))
            print("\n")
        title_input = input("Please enter a movie title (type exit() to exit):\n")
    print("\n")

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://locahost", PORT=9200)
    mainLoop1(es)