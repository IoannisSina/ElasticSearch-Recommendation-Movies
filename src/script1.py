"""
Read movies.csv and add the to Elasticsearch DB
"""
import os
from elasticsearch import Elasticsearch, helpers
import pandas as pd

def bulk_json_data(movies, _index):
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
        "size": 10,
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

if __name__ == "__main__":
    es = Elasticsearch(HOST="http://locahost", PORT=9200)
    # insert_movies(es)
    response = search_movies(es, "Toy Story")
    print(response)