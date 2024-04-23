import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer #new work
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
file_path = os.path.join(parent_dir, 'resources/processed_emails.csv')
#print(file_path)
emails_df = pd.read_csv(file_path)
print(emails_df.columns)

#input and target data
X = emails_df['tokens']
Y = emails_df['spam']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

#vectorize data (if lost, consult previous naive-baye project (movies one))
vectorizer = TfidfVectorizer()
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

def get_avg_cv_means(alphas):
    avg_cv_scores = []
    for alpha in alphas:
       naive_baye_classifier = MultinomialNB(alpha=alpha)
       kf = KFold(n_splits=5) 
       cv_scores = cross_val_score(naive_baye_classifier, x_train_vectorized, y_train, cv = kf)
       avg_cv_scores.append(np.mean(cv_scores))
    return  avg_cv_scores

def plot_cv_mean_accuracy(alphas,avg_cv_scores):


    plt.figure(figsize=(10,6))
    plt.plot(alphas, avg_cv_scores, marker='o', color='purple')
    plt.title("CV mean Accuracy scores vs Alpha Parameter values")
    plt.xlabel('Alpha value')
    plt.ylabel("CV Mean Accuracy score")
    plt.grid()
    plt.show()
    return
alphas = np.arange(0.08, 0.1, 0.001)
plot_cv_mean_accuracy(alphas, get_avg_cv_means(alphas))

#get the alpha value with the maximum cv avg score
best_alpha_index = np.argmax(get_avg_cv_means(alphas))
best_alpha = alphas[best_alpha_index]
print(f"Best alpha value: {best_alpha}")

#train model on best alpha value and evaluate accuracy
Alpha = best_alpha
naive_baye_classifier = MultinomialNB(alpha=Alpha)
naive_baye_classifier.fit(x_train_vectorized, y_train)

#make predicition
y_pred = naive_baye_classifier.predict(x_test_vectorized)

print(f"Accuracy: {accuracy_score(y_test, y_pred)*100}")
print("classification report:")
print(classification_report(y_test, y_pred))

