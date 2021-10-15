# Description :zap:
In this project movies are inserted in [ElasticSearch](https://www.elastic.co/) in order to create a movie recomendation system. The user enters a string (movie title) and ElasticSearch returns the movies whose title is most similar to that string, based on the [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) metric. Movies scores are modified using Kmeans and Neural Networks based on the user's ratings. A non existing rating is predicted by the models created.

Before using any of the script, you must enter a valid ID **(1 - 671)**.

### Script1.py
In this script are inserted to ElasticSearch and a function in created to retrieve similar movies.

### Script2.py
In this script, similar movies are retrieved, and their score is updated based on the user’s rating. An array with the score of each movie is printed in the terminal.

### Script3.py
In this script [Kmeans](https://en.wikipedia.org/wiki/K-means_clustering) algorithm is used to cluster users based on their genre ratings. When a user has not rated a movie, the rating is calculated using the mean rating from the cluster that the user belongs to. An array is printed indicating the movies’ scores after the prediction of the rating.

### Script4.py
In this script a Neural Network is created using [Keras](https://keras.io/) for the current user based on his/her genre ratings. Movie titles are converted to vectors using the [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) technique and are used as an input to the created model. After training the model, any non-existing rating can be predicted. An array with the score of each movie is printed after the model predictions.

# Steps to run 🏃
1. Download and Install ElasticSearch [here](https://www.elastic.co/downloads/elasticsearch).
2. To run ElasticSearch navigate to the folder and run bin/elasticsearch.
3. Now download the repository and navigate to that folder.
~~~
git pull https://github.com/IoannisSina/ElasticSearch_recommendation_movies
cd src
~~~
4. Install the required libraries.
~~~
pip install -r requirements.txt
~~~
5. From script1.py run the **insert_movies** function to insert the movies to ElasticSearch.
6. Finally choose the desired script and run it following the steps shown in the terminal.
