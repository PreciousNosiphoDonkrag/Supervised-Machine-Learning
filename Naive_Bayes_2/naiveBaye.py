#run download_resources.py
#then run datapreperation.py
#only then can you run this file
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #new work
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, 'resources/processed_emails.csv')
emails_df = pd.read_csv(file_path)
#print(emails_df.columns)

#input and target data
X = emails_df['tokens']
Y = emails_df['spam']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#vectorize data (if lost, consult previous naive-baye project (movies one))
vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

#train model
Alpha = 0.4
naive_baye_classifier = MultinomialNB(alpha=Alpha)
naive_baye_classifier.fit(x_train_vectorized, y_train)

#make predicition
y_pred = naive_baye_classifier.predict(x_test_vectorized)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100}")
print("classification report:")
print(classification_report(y_test, y_pred))


