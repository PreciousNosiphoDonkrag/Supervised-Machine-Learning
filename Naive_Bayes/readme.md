# Naive Bayes Classifier
## what is the project about
## Working with new functions from the Sklearn library
from sklearn.naive_bayes import MultinomialNB <br><br>
- **Sklearn.naive_bayes** is a module that has several classes for implementing      naive baye classification. <br><br>
- **MultinomialNB** class is one of these implementations, it is designed for     multinomially distributed data, which is commonly encountered in text       classification tasks. <br>
**What is multinomial distributed data** <br>
  multinomial distribution describes the probability of observing each possible      outcome in a fixed number (n) of independent trials, where each trial results      in one of several mutually exclusive outcomes. <br>

- By using MultinomialNB, we can train a Naive Bayes classifier specifically         tailored for handling multinomially distributed features, such as word counts in   text data. This classifier calculates the probability of each class given a set    of features and predicts the class with the highest probability. <br>

from sklearn.feature_extraction.text import TfidfVectorizer #new work


Vectorizing the text data is a crucial step in preparing textual data for machine learning algorithms, including Naive Bayes classifiers. Here's why we vectorize the text data:

Numerical Representation: Machine learning algorithms work with numerical data. Vectorization converts textual data (plot keywords) into numerical feature vectors, making it suitable for algorithms to process.

Feature Extraction: Vectorization helps in extracting relevant features from text data. In the context of TF-IDF vectorization, it assigns weights to each word based on its importance in a document relative to the entire corpus. This helps in capturing the importance of words while considering their frequency across all documents.

Dimensionality Reduction: By converting text data into numerical vectors, vectorization reduces the dimensionality of the feature space. This is essential for improving computational efficiency and preventing issues like the curse of dimensionality.

Semantic Similarity: Vectorization captures the semantic similarity between words. Words with similar meanings tend to have similar vector representations, which can improve the model's ability to generalize.

Overall, vectorizing text data is a fundamental preprocessing step that transforms raw textual data into a format suitable for machine learning algorithms, enabling them to learn patterns and make predictions effectively. In the case of Naive Bayes classifiers, vectorization allows the algorithm to work with text data and make predictions based on the extracted features.
