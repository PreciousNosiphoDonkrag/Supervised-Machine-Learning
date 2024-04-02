
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import os
from image_preperation import prepare_input_image
from split_data import MNIST_data
#get the location of the csv
current_dir = os.path.dirname(os.path.realpath(__file__))


#load the data
#notice we do not split the data because it is already seperated 
#into traning and testing data 
#train_test_split will not be used
X_train, y_train, X_test, y_test = MNIST_data()
#print(X_train.shape, y_train.shape,X_test.shape,y_test.shape)

def flatten_X_data(data):
    return data.reshape(data.shape[0], -1)


#flattening x data: 60000 x 28 x 28 and 10000 x 28 x 28 
X_train_flat = flatten_X_data(X_train) #shape is 60000 X 784
X_test_flat = flatten_X_data(X_test) #shape is 10000 x 784

class knn:
    def __init__(self, k=5):
        self.k = k
        self.model = None

    def fit(self, xtrain, ytrain):
        
        self.model = KNeighborsClassifier(n_neighbors=self.k)
        self.model.fit(xtrain, ytrain)
        
    def predict(self,xtest):
        return self.model.predict(xtest) #return y prediction
    
    def evaluate_accuracy(self, xtest, ytest):
        
        y_predict = self.predict(xtest)
        accuracy = accuracy_score(ytest, y_predict)
        return accuracy
    
#Part 1: Train model on the training data and evalute accuracy on the testing data
#--------------------------------------------------------------------------------------
# def knn_initator():
#     knn_object = knn()
#     #fit the object with the training data
#     knn_object.fit(X_train_flat, y_train)
#     Accuracy = knn_object.evaluate_accuracy(X_test_flat, y_test)
#     print(f"The accuracy score for the testing data is\t{round(Accuracy*100,2)}%")
# knn_initator()

#part 2:Train model on the training data and evalute accuracy on the independent images
#--------------------------------------------------------------------------------------
def get_accuracy_point(k=5):
    #Create an instance of the knn class
    knn_object = knn(k)

    # #fit the object with the traning data
    knn_object.fit(X_train_flat, y_train)
    print(knn_object.evaluate_accuracy(X_test, y_test))
    #get the accuracy value for the independent images in the images of numbers folder
    counter = 0 #for accuracy
    for n in range(10):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        images_dir = os.path.join(current_dir, "images_of_numbers")
        image_path = os.path.join(images_dir, f"{n}.JPG")
        
        #prepare the image
        prep_img = prepare_input_image(image_path)
        
        # reshape the image to (1,784)
        img_reshaped = prep_img.reshape((1,-1,))
        
        prediction = knn_object.predict(img_reshaped)
        if(n == prediction[0]):
            counter = counter + 1
    accuracy_score2 = (counter/10)*100
    return accuracy_score2


#plot the accuracy for different k values use the k value that gives the highest accuracy 
def plt_accuracy():
    k_values = list(range(1, 10))  # Adjust the range as needed
    accuracy_values = []

    for k in k_values:
        accuracy = get_accuracy_point(k)
        accuracy_values.append(accuracy)

    # Find the k value with the highest accuracy
    best_k = k_values[np.argmax(accuracy_values)]
    # Plot the results
    plt.plot(k_values, accuracy_values, marker='o')
    plt.title('Accuracy for Different k Values')
    plt.xlabel('k Value')
    plt.ylabel('Accuracy')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

    return best_k


#Predict the independent images in the images of numbers folder
def predict_images():
    
    #Create an instance of the knn class
    knn_object = knn(plt_accuracy())

    # #fit the object with the traning data
    knn_object.fit(X_train_flat, y_train)
    #get accuracy for independent images not in dataset
    print("Predicting images in images of numbers folder")
    print("-------------------------------------------------")
    counter = 0 #for accuracy
    for n in range(10):
        #C:\Users\nosip\Desktop\Supervised-Machine-Learning\KNN\project\images_of_numbers
        images_dir = os.path.join(current_dir, "images_of_numbers")
        image_path = os.path.join(images_dir, f"{n}.JPG")
        #image_path = f"./images_of_numbers/{n}.JPG"
        
        #prepare the image
        prep_img = prepare_input_image(image_path)
        
        # reshape the image to (1,784)
        img_reshaped = prep_img.reshape((1,-1,))
        
        prediction = knn_object.predict(img_reshaped)
        
        if(n == prediction[0]):
            counter = counter + 1
    
        print(f"prediction of {n} \t is \t {prediction[0]}")
    accuracy_score2 = (counter/10)*100  
        
    print(f"\n Predicts independent images with a accuracy of:\t {accuracy_score2}%")
 
predict_images()

