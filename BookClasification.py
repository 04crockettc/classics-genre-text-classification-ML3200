import numpy as py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Load the goodreads dataset
data = pd.read_csv("goodreads_data.csv")
#print(data.head())

# features in dataset = [number, book, author, desription,
# genres, avgRating, numRating, URL]
# creates a binary target. 
data["is_classic"] = data["Genres"].str.contains("classics", case = False, na = False).astype(int)


# clean the text: handle missing discriptions, change strings to lower case, and remove puncutation
data["Description"] = data["Description"].fillna("")
data["Description"] = data["Description"].str.lower()

#split the dataset into train and test
x = data["'Description"]
y = data['is_classic']


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state = 42
)

#TF-IDF 
#Convert a collection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train['Description'].fillna(''))
X_test_tfidf = vectorizer.transform(X_test['Description'].fillna(''))
vectorizer.get_feature_names_out()

#print(X_train_tfidf.shape)
#print(X_test_tfidf.shape)


#fit the model on training data
#test the model
#evaluate the results