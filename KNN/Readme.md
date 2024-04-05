## K-Nearest Neighbors (KNN) Algorithm
The KNN algorithm is a lazy, non-parametric supervised learning algorithm used for classification of data. The KNN algorithm does not process data until it needs to make a prediction (classify it); thus KNN does not make an assumption about the dataset or attempt to understand the data and it does not build a model for the dataset, it just classifies it during the prediction phase. 
A short overview of how KNN works is given below:
![image](https://github.com/PreciousNosiphoDonkrag/Belgium_ITVersity_Campus_Studies/assets/153648767/98df9c32-7e33-488f-bf50-c8e3c8b51107) <br>
For a given input (X), the algorithm find “K” training points closest to X, and makes a prediction based on these points. Thus, KNN algorithm uses the distances between the input value X and all the available training data points.
For the above example, we have 2 classes (blue and red), if we take K = 3, the closest 3 points to X will be used to predict the class of X.

**numbers that appear more than once (5, 6) are the same distance away from X.**<br>
**For K = 3:<br>**
The closest 3 points are 1, 2 and 3. <br>
2/3 points are blue which means we predict X to be blue.<br><br>
**For K = 7:<br>**
5/9 points are red, so we predict X to be red.<br>
**notice that for K = 5 and K = 6 we have a tie**<br><br>
To break a tie the following methods can be applied:<br><br>

 **Weighted Voting (for Classification):**<br>
If you use weighted voting, each neighbor's vote is given a weight based on its proximity to the input point. This means closer neighbors have a stronger influence on the prediction. In case of a tie, the class of the closer neighbor may be given more importance.<br><br>
**Distance as a Deciding Factor (for Regression):**<br>
In regression tasks, when there is a tie in the predicted values, you can consider the distance of each neighbor as a deciding factor. The value from the closer neighbor may be given more weight in such cases.<br><br>
**Random Selection:**<br>
Another approach is to break ties randomly. For example, if you have a tie between two classes, you can randomly select one of them as the predicted class.<br><br>
### The goal for this section is to create an image classifier for handwritten digits (0-9) using the KNN algorithm.<br> 

### The approach:<br>
The logic behind KNN was easy to grasp, however the implementation was challenging. Hence to meet the goal, this section will be broken down as following:<br>

  * Learning the mathematical implementation of KNN model in Python using @NeuralNine blueprint explained in the video “K-Nearest Neighbors Classification From Scratch in Python (Mathematical)”. <br> - Link: https://www.youtube.com/watch?v=xtaom__-drE&t=1116s <br><br>
*	Introduce a new dataset, thus allowing for data visualization. <br><br>
*	Build on his implementation by adding training data in order to implement performance monitoring of the algorithm and introduce sklearn-Learn. <br><br>
*	Explore sklearn-learn KNN implementation <br><br>


