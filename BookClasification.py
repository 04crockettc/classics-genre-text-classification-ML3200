import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

#Load the goodreads dataset
data = pd.read_csv("goodreads_data.csv")
# features in dataset = [number, book, author, desription,
# genres, avgRating, numRating, URL]

# creates a binary target. 
data["is_classic"] = data["Genres"].str.contains("classics", case = False, na = False).astype(int)


# clean the text: handle missing discriptions, change strings to lower case, and remove puncutation
data["Description"] = data["Description"].fillna("")
data["Description"] = data["Description"].str.lower()


#split the dataset into train and test
x = data["Description"]
y = data["is_classic"]


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state = 42
)

#TF-IDF 
#Convert a collection of raw documents to a matrix of TF-IDF features
vectorizer = TfidfVectorizer(stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
#print(X_train_tfidf.shape)
#print(X_test_tfidf.shape)



#Fit model using logistic regression
logistic_X_train_tfidf = X_train_tfidf
logistic_y_train = y_train
logistic_y_test = y_test
logistic_X_test_tfidf = X_test_tfidf

model = LogisticRegression(max_iter= 500)
model.fit(logistic_X_train_tfidf, logistic_y_train)

#predict using test set ofr logistic regression
logistic_y_pred = model.predict(logistic_X_test_tfidf)


# fit model to linear svc
model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# predict linear svc using the test set
y_pred = model.predict(X_test_tfidf)


# evalutate performance
print("Test Accuracy for Logistic Regression:", accuracy_score(logistic_y_test, logistic_y_pred), "\n")
print(classification_report(logistic_y_test, logistic_y_pred))

print("Test Accuracy for LinearSVC:", accuracy_score(y_test, y_pred), "\n")
print(classification_report(y_test, y_pred))

# Visualize results using matplotlib
#accuracy comparisons between models
models = ["Logistic Regression","LinearSVC"]
accuracies = [accuracy_score(y_test, logistic_y_pred), accuracy_score(y_test, y_pred)]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies, color=['green','orange'])
plt.ylim(0.6,1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

#Class ditribution 
classes, counts = np.unique(y, return_counts=True)
plt.figure(figsize=(6,4))
plt.bar(["Non-Classic","Classic"], counts, color=['skyblue','green'])
plt.title("Distribution of Classic vs Non-Classic Books")
plt.ylabel("Number of Books")
plt.show()

