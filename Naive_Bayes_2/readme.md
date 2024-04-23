# Naive_baye Classifier: 
## Spam Emails detection
This project aims to classify emails as spam or non-spam (ham) using the Naive Bayes classifier. The project consists of three main components:

### Data Preparation (data_preperation.py):
- Cleans and preprocesses the email dataset.
- Performs text normalization, tokenization, stopwords removal, stemming, 
- lemmatization, and removes special characters and punctuation.
### Naive Bayes Classifiers:
- Uses the processed dataset to train a Naive Bayes classifier.
- Vectorizes the text data using TF-IDF
- Utilizes cross-validation to find the best alpha value for the Naive Bayes classifier.
- Trains the Naive Bayes classifier with the best alpha value and evaluates its accuracy.
### Download Resources (download.py):
Downloads necessary resources such as stopwords and WordNet data required for data preprocessing.
## Parameters in ML
A  parameter is an internal variable that can be configured to the ML model
and whose value is estimated from the training data. These parameters are used to make predictions
on the new data.
### Model parameters: 
- These are parameters that are learned from the training data during the training
   process.
- They represent the internal state of the model and are adjusted automatically
  by the learning algorithm.
- Example: The coefficients in a linear regression model (intercepts too). 
### Hyperparameters (Tuning Parameters):
- These are configuration settings external to the model that cannot be learned
  from the data.
- They are set before training and remain constant during training.
- Hyperparameters control aspects of the learning process and model complexity.
- Example: the alpha parameter for naive-Baye classifier (will be discussed below)

### Alpha parameter in Multinomial Naive Baye Classifer
 the alpha parameter (commonly denoted as α) is a smoothing parameter used for 
 additive (Laplace) smoothing.
 #### Laplace smoothing
 - Used to handle cases where a word in the test data did not appear in the training
   data; this is to avoid zero probabilities for features not present in the training data.
   - When alpha = 0; no smoothing is applied.
   - Effect of Alpha Parameter:
     1. A larger alpha value results in more smoothing.
     2. Smaller values of alpha mean less smoothing.
     3. A value of alpha = 1 is often used as a starting point.
    
#### Setting Alpha:
- The alpha parameter is set when creating an instance of the Multinomial Naive Bayes classifier.
 ![Screenshot 2024-04-22 124320](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/06caff15-d561-4be0-aff9-a0aee7560480)

- The optimal value of alpha can be found through parameter tuning techniques such as grid search or cross-validation; both of these
- tuning techniques will be discussed under their own folder.       

   
## What i learned
- ML models' parameters and parameter tuning (alpha parameter)
- Cross-validation to tume parameters
