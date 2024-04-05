## Understanding the Iris Dataset<br>
The Iris dataset is found in the scikit-learn python library and it is widely used for machine learning models.<br>
The dataset consists of 150 observations of the iris flowers and 3 different types of irises'.<br> 
Each row in the dataset represents a flower. <br>

the following columns exist in the dataset:<br>
  *  Sepal Length <br>
  *  Sepal Width <br>
  *  Petal Length <br>
  *  Petal Width <br>
  *  Species of flower <br>

  The **iris_data** is a feature matrix (input data) from the iris data and it'll be assigned to the variable x.<br>
  The **iris_target** is the target array and it contains classes the iris_data belong to. This is what we will try to predict from a random input X. The dimensions of the flower will be the input, and the algorithm will predict which class it belongs to in Y. <br> 
  
  ### Layout of data <br>
  ![layout](https://github.com/PreciousNosiphoDonkrag/Belgium_ITVersity_Campus_Studies/assets/153648767/338ac87e-82a7-43d3-aad1-99ebd6701639) <br>

  ### Graphical Visualization of data
  ![gggg](https://github.com/PreciousNosiphoDonkrag/Belgium_ITVersity_Campus_Studies/assets/153648767/c0145bfb-236e-4e63-9f92-bc83aad65a54)

### The Layout of the code
The following code builds on the mathematical implementation of the previously implemented KNN algorithm. however, it replaces the matrix of coordinates that was previously used with the iris dataset. This has allowed for a smooth introduction to the scikit-learn library. The layout of the code follows the basic outline of machine learning:<br>
- Data Preperation: The iris data set is split into X and Y training and testing data.<br>
-  model Instatiation: the KNN class is still there however inside the class the model is instantiated using the   KNeighborsClassifier found in scikit-learn, this is done inside the fit method.
-  Model Training: still inside the fit method, the model is trained using the X and Y training data, that was defined in data preparation. <br>
- Make a prediction: The prediction is made on the X training data, and the predict method returns an array of y predictions.<br>
- Performance evaluation (this is new): To test the performance of the model, the returned Y prediction is tested against the Y test data to produce an accuracy score. the scikit-learn model produced an accuracy score of 1.<br>

### Some images if you are interested
![knnP1](https://github.com/PreciousNosiphoDonkrag/Belgium_ITVersity_Campus_Studies/assets/153648767/513601a5-c614-4564-9893-f40e767e7cf6)
![knnp2](https://github.com/PreciousNosiphoDonkrag/Belgium_ITVersity_Campus_Studies/assets/153648767/8cf9efa5-b5a5-48aa-a605-93c1943e6c4e)

***This section highlighted the benefits of using the scikit-learn pythin library, it produced an efficient code because of all the readily available methods and classes at our disposal.*** <br>  


