# Cross-validation (cv)
Cross-validation is a technique used to assess how well a model will 
generalize to an independent dataset. It involves splitting the training data 
into multiple subsets, training the model on a subset of the data, and evaluating it on the remaining subset. This process is repeated multiple times, 
with each subset used as the validation data exactly once.

## How it works (Diagram from scikit-learn.org)
![Screenshot 2024-04-22 144938](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/3e1d0411-cabf-4c9c-b2fc-709fb9cd86ee)
- The dataset is split into training and testing data.
- Then the training data is further split into k folds.
- The chosen model is trained k number of times.
- Cross-validation ensures that each fold is treated as the training and the testing data at least once.
- Once this is done there will be k different models and metrics.
- To get the final performance metric, you typically average the k performance metrics obtained from each iteration.
- Once you've completed the k iterations, you can test your final model on the untouched test dataset to get an estimate of how
- well your model will perform on unseen data.

  ## Code
  - Perform a 5-fold cv to get familar with the cross_val_score() function:
  ![Screenshot 2024-04-23 113003](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/7b49144f-d9fd-40f7-af0d-838ab0bc530e)
    - The **kFold** object is used to split the data into 'k' consecutive folds. It can be
      adjusted to shuffle the folds so they are not consecutive.
    - The **cross_val_score()** function is used to perform cv and it comes from the sk-learn library.
      It takes in the model (naive_baye_classifier), the training data, and the number of folds.
    - **cv=kf** Tells scikit-learn to use the KFold object 'kf' created as the cross-validation splitting strategy.  
    - The cross_val_score() return the cv scores that are then averaged to get the mean accuracy score.
   
  - The mean accuracy score tells us how well the model will generalized to unseen data; **it does not give the actual accuracy
    of the trained model**. Then why bother with CV??    
