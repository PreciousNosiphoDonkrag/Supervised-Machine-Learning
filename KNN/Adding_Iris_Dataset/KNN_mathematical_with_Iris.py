import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# The Aim of this project is to give a mathematical explanation of K-Nearest Neighbors

#1. Data Preperation

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
species_names = ["setosa", "versicolor", "virginica"]

# Split the data into training and testing sets
#test_size=0.2: specifies that 20% of the data will be used for testing, and the remaining 80% will be used for training.
#random_state=42: Setting the random state to a fixed number ensures that the random split will be the same every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#2. model Instantiation and Training
# Define a class that will implement KNN
class KNN:
    def __init__(self, k=5):  # Constructor with default k value
        self.k = k
        #When an instance of the KNN class is created, 
        # the __init__ method is called automatically. 
        # In this method, self.model is initialized to None. 
        self.model = None #create an attribute that will hold the instsnce of the model

    def fit(self, X_train, y_train):
        #create an instance of KNeighborClassifier class found in scikit-learn
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        
        #calling fit on the knn instance creates a KNeighborsClassifier
        #model with specified number of neighbors and fits it to
        #the training data
        self.model.fit(X_train, y_train)

#3. Make a prediction

    #predict returns an array because in scikit-learn it is designed to
    #return a prediction for multiple values
    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate_accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy

    #Lets map the species name to the integers of the classes
    def map_species_name(self, class_Name):
        Name = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        return Name[class_Name]
    
# Instantiate KNN classifier
knn_classifier = KNN()

# Fit the classifier on the training data
knn_classifier.fit(X_train, y_train)

#Add an input point to predict
#notice that the input point is a 2D aaray instead of a 1D array with 2 elements
#This is done to maintain consistency because the predict function
#from scikit-learn expects a 2D array
input_point = np.array([[5.5, 3.0,  2.0, 0.3]]) #adjust this point to see predictions
predict_class = knn_classifier.predict(input_point)
species_Name = knn_classifier.map_species_name(predict_class[0])
print(f"The predicted class for the provided input is: {species_Name}")

# Evaluate the accuracy on the test set
accuracy = knn_classifier.evaluate_accuracy(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")



# Visualization
ax = plt.subplot()
ax.grid(True, color="#323232")
ax.set_facecolor("#FFFFFF")
ax.figure.set_facecolor("#FFFFFF")
ax.tick_params(axis="x", color="black")
ax.tick_params(axis="y", color="black")

# Plot existing points
colors = ["#02C04A", "#FF0000", "#0000FF"]
for i in range(3):
    indices = np.where(y == i)
    ax.scatter(X[indices, 0], X[indices, 1], color=colors[i], s=60, label=f"Class {species_names[i]}")


#plot the input point to predict
ax.scatter(input_point[0, 0], input_point[0, 1], color='black', marker='*', s=200, label=f"Input Point\nPredicted Class: {species_Name}")

ax.set_xlabel(iris.feature_names[0])
ax.set_ylabel(iris.feature_names[1])

# Show the plot
plt.legend()
plt.show()

