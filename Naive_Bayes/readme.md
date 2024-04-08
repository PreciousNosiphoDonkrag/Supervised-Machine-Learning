# Naive Bayes Classifier
## what is the project about
## Working with new functions and modules from the Sklearn library
- **Sklearn.naive_bayes** is a module that has several classes for implementing      naive baye classification. <br><br>
- **MultinomialNB** class is one of these implementations, it is designed for     multinomially distributed data, which is commonly encountered in text       classification tasks. <br><br>
**What is multinomial distributed data** <br>
  multinomial distribution describes the probability of observing each possible      outcome in a fixed number (n) of independent trials, where each trial results      in one of several mutually exclusive outcomes. <br>

- By using MultinomialNB, we can train a Naive Bayes classifier specifically         tailored for handling multinomially distributed features, such as word counts in   text data. This classifier calculates the probability of each class given a set    of features and predicts the class with the highest probability. <br>

## TfidfVectorizer
This class is used to convert a collection of raw text documents into a matrix of Term Frequency- Inverse Document Frequency (TF-IDF).
- TF-IDF: this is a number (weight) that reflects the importance of a word relative to the whole document/s. It is calculated as follows:
  ![Screenshot 2024-04-08 193757](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/2ccb405d-91f1-44b0-bfaf-5daf956f69ba)
- The TF-IDF model considers both the unique and rare words in its calculations. This is to accommodate for common words such as the, is and etc.
- TfidfVectorizer converts each document into a **number vector** based on the TF-   IDF scores; <br> 
- Each dimension of the vector corresponds to a **unique word** in the vocabulary of     the document; <br>
- and the value of each dimension represents the TF-IDF score of the corresponding   word in the document. <br>
![Screenshot 2024-04-08 204035](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/e950166e-2836-4887-8496-25a86df89bab)
    - The word "the" in the above image is common, hence its TF-IDF score will not be calculated by the TfidfVectorizer.
 
- **Normalization:** After vectorization, the TF-IDF vectors are often normalized to ensure that each document vector has a unit norm (length). Normalization can prevent longer documents from dominating the similarity calculations.

## The code

