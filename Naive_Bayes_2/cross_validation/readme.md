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
- Once this is done there will be k different metrics.
- To get the final performance metric, you typically average the k performance metrics obtained from each iteration.
- Once you've completed the k iterations, you can train your final model on the best parameter value (alpha).

  ## New functions:
  - Perform a 5-fold cv to get familar with the cross_val_score() function:
  ![Screenshot 2024-04-23 113003](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/7b49144f-d9fd-40f7-af0d-838ab0bc530e)
    - The **kFold** object is used to split the data into 'k' consecutive folds. It can be adjusted to shuffle the folds so they are not consecutive.
    - The **cross_val_score()** function is used to perform cv and comes from the sk-learn library.
      It takes in the model (naive_baye_classifier), the training data, and the number of folds.
    - **cv=kf** Tells Scikit-learn to use the KFold object 'kf' created as the cross-validation splitting strategy.  
    - The cross_val_score() returns the cv scores that are then averaged to get the mean accuracy score.
   
  - The mean accuracy score tells us how well the model will generalized to unseen data; **it does not give the actual accuracy of the trained model**.

## The code
- A range of alpha values is generated and stored in an array.
- This array is sent to *get_avg_cv_means()* where:
  - An instance of the Multinomial Bayes-classifier is created for each alpha value;
  - CV is then performed for each alpha value (number of folds = 5)
  - The scores corresponding to each alpha value are averaged and stored in *avg_cv_scores*
  - These alpha values and their corresponding cv average scores are plotted.
    - The first plot:
      ![Screenshot 2024-04-23 165428](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/d0be2fcb-7f91-43c6-a995-cf8417874133)
      - From the diagram it is drawn that the model generalizes best for an alpha value of 0.1.
    - Changing the range of alpha values
     ![Screenshot 2024-04-23 165702](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/a70e17f3-1b61-4ed2-9272-5cd5000e84bb)
![Screenshot 2024-04-23 214033](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/2856ece3-78a3-48d4-93d0-0cb47090e2d0)

    - Playing around with the ranges of the alpha values to get the highest CV mean accuracy score; did not significantly change the actual accuracy score of the model once it was trained.

The accuracy score of the model alternated between 98.156% and 98.332%.
This is a significant improvement from 94.644% without using cross-validation to 98.156% while using cross-validation. The best alpha value to give an accuracy of 98.156% is 0.08. 

## General guideline for choosing k:
- Small to Medium Datasets (less than 10,000 data points): You can use k values between 5 and 10.
- Medium to Large Datasets (more than 10,000 data points): You can use larger values of k, such as 10, 15, or even 20.

