#'plot_keywords'
#'movie_title'
#'imdb_score'
import numpy as np
from dataPreperation import start_here
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #new work
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

movies_df = start_here()
#print(movies_df.head())

X = movies_df['plot_keywords'].str.split('|').apply(lambda x: ' '.join(x))  # Extract plot keywords
Y = movies_df['quality']

#Split the data into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#vectorize the data to:
#1. convert text to numeric data
#2. helps in capturing the importance of words while
#   considering their frequency across all documents
#3. Dimension reduction
#4. Vectorization captures the semantic similarity between words. 
vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#Evaluate model
naive_bayes_classifier = MultinomialNB() #use multinomial because the target data can take on more than 2 discrete values
naive_bayes_classifier.fit(x_train_vectorized,y_train)

#make a prediction
y_pred = naive_bayes_classifier.predict(x_test_vectorized)
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100}")
print("classification report:")
print(classification_report(y_test, y_pred))
