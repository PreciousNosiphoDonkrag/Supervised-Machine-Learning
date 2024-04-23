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
x_vectorized = vectorizer.fit_transform(X)

#Define model
Alpha = 0.4
naive_baye_classifier = MultinomialNB(alpha=Alpha)

#perform n-fold cross validation
no_folds = 5
kf = KFold(n_splits=no_folds) #split data into k-consecutive folds; can change setting to shuffle
#cv=kf : Tell scikit-learn to use the KFold object kf 
#created as the cross-validation splitting strategy.
cv_scores = cross_val_score(naive_baye_classifier, x_vectorized, Y, cv = kf) 

print(f"Cross Validation scores \n {cv_scores}")
print("classification report:")
print(f"Mean of scores: {np.mean(cv_scores)}")

#Lets plot some accuracy scores
accuracy_scores = []
k_values = range(5, 11)
for k in k_values:
  kf2 = KFold(n_splits=k)
  cv_scores = cross_val_score(naive_baye_classifier, x_vectorized, Y, cv = kf2)
  mean_accuracy = np.mean(cv_scores)  
  accuracy_scores.append(mean_accuracy)

plt.figure(figsize=(10,6))
plt.plot(k_values, accuracy_scores, marker='o')
plt.title("Mean Accuracy scores vs K values")
plt.xlabel('Number of folds (k)')
plt.ylabel("Mean Accuracy")
plt.show()

#get the k-value with the highest accuraacy between 5 and 10
best_k = k_values[np.argmax(accuracy_scores)]
highest_accracy = np.max(accuracy_scores)
print(f"best k-value: {best_k}\tHighest accuracy: {highest_accracy}")
#With k-fold cross-validation, the dataset is automatically
#split into k folds, and the model is trained and evaluated
# k times, each time using a different fold as the test set
# and the remaining folds as the training set.

#So, there's no need to manually split the data into training
# and testing sets


###Here's a general guideline for choosing the value of k:
#Small to Medium Datasets (less than 10,000 data points):
#You can use k values between 5 and 10.
#For example, you might use 5-fold or 10-fold cross-validation.
#Medium to Large Datasets (more than 10,000 data points):
#You can use larger values of k, such as 10, 15, or even 20.
#For example, you might use 10-fold, 15-fold, or 20-fold cross-validation.
