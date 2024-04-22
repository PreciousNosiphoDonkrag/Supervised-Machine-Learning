# Naive_baye Classifier: 
## Spam Emails detection
## Parameters in ML
A  parameter is an internal variable that can be configured to the ML model
and who's value is estimated from the training data. there parameters are used to make predictions
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
- Example: the alpha parameter for naive-Baye classifier (will be discussed later)

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

- The optimal value of alpha can be found through parameter tuning techniques such as grid search or cross-validation       
## Cross validation
Cross-validation is a technique used to assess how well a model will 
generalize to an independent dataset. It involves splitting the training data 
into multiple subsets, training the model on a subset of the data, and evaluating it on the remaining subset. This process is repeated multiple times, 
with each subset used as the validation data exactly once.

### How it works (Diagram from scikit-learn.org)
![Screenshot 2024-04-22 144938](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/3e1d0411-cabf-4c9c-b2fc-709fb9cd86ee)
- The dataset is split into training and testing data.
- Then the training data is further split into k folds.
- The chosen model is trained k number of times.
- Cross-validation ensures that each fold is treated as the training and the testing data.
- Once this is done there will be k different models and metrices
- To get the final performance metric, you typically average the k performance metrics obtained from each iteration.
- Once you've completed the k iterations, you can test your final model on the untouched test dataset to get an estimate of how well your model will perform on unseen data.
- In scikit-learn the **cross_val_score()** function takes the model, input features,
  target labels, and the number of folds as input and returns the cross-validation scores.
   
## What i learned
- Introduction to ML models' parameters and parameter tuning
