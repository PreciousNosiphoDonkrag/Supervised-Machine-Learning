# Handwritten Digit Recognition with K-Nearest Neighbors (KNN)

This project aims to implement a K-Nearest Neighbors (KNN) algorithm for 
recognizing handwritten digits. The implementation utilizes the MNIST dataset for 
training and testing the model's performance, as well as personal handwritten 
images provided under the folder: images_of_numbers.

## About the MNIST dataset
The MNIST dataset is a collection of images of handwritten digits (0 through 9),
where each image is a grayscale image of size 28x28 pixels. 
The dataset is commonly used for training various machine learning models for image classification tasks.

### The structure of the Dataset
  - The 1st column of the dataset is the digit value; this is the target 'yvariables used for testing the model.
  - The 2nd column to the last, makes up the pixel values for each image. the pixel value can take
   on any value between 0 - 255.
  - Thus, each row in the dataset represent 1 image. <br>
![Screenshot 2024-04-02 220921](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/ee2f8b12-f9b1-488d-94a5-bb8c17ec5b7d)

The dataset will be split 80-20. where 80% of the data is used for training and 20% is used for testing.

## Overview

The project consists of the following files:

1. **showcase_MNIST_images.py**: This file plots the first 10 images in the dataset, so the user can be familar with how they look. 

2. **split_data.py**: This file splits the MNIST dataset into training and testing sets, preparing it for use in the KNN algorithm.

3. **image_preperation.py**: This file provides functions for preprocessing user-provided handwritten images, ensuring they meet the requirements for compatibility with the MNIST dataset and the KNN algorithm.

4. **MNIST_KNN.py**: This file contains the main implementation of the KNN algorithm, including loading the MNIST dataset, training the model, and evaluating its performance on both the MNIST dataset and personal handwritten images.

## Image preperation requirements
The image must meet the following requirements to ensure data consistency
* Grayscale (the MNIST images are all in gray)
* Size: 28x28
* ensure that the handwritten number is clearly visible
* with a constransting background (black pen white page)
* scale the pixels to a range between [0,1]
## Usage

1. **Training the Model**: Run `MNIST_KNN.py` to train the KNN model using the MNIST dataset. The model will be trained on the training data and evaluated on the testing data to assess its performance.

2. **Testing with Personal Handwritten Images**: After training the model, you can provide your own handwritten images in the `images_of_numbers` folder. Run `MNIST_KNN.py` again to test the model's performance on these images. The model will predict the labels for each image and compare them to the actual labels to calculate accuracy.

3. **Plotting Accuracy for Different k Values**: The `plt_accuracy()` function in `MNIST_KNN.py` plots the accuracy of the model for different values of k. This helps determine the optimal value of k for the KNN algorithm.

4. **Predicting Independent Images**: Finally, the `predict_images()` function in `MNIST_KNN.py` uses the best value of k obtained from the accuracy plot to predict the labels of your handwritten images. It prints the predictions along with the accuracy score.

## Output
Images from the dataset:
![Screenshot 2024-04-02 223310](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/96ebb8bd-b04b-4e42-bf6d-018be8b9ad0a)

Personal hand writtien digits before and after image preperation but before flattening pixels into a 1D array
![Screenshot 2024-04-02 231232](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/8f9bd91e-a2c6-4b17-9b1a-5609fe846e91)
![Screenshot 2024-04-02 231303](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/de553ef2-03b0-49bb-bad6-ffc81b3b0220)
![Screenshot 2024-04-02 231251](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/52bf84f4-60f1-4772-9d2a-639b191ff0ac)

### Accuracy plot for different K-vaules
![Screenshot 2024-04-02 230726](https://github.com/PreciousNosiphoDonkrag/Supervised-Machine-Learning/assets/153648767/cc35d0b4-122b-42a3-9f37-52fc90f13167)

- On the homogeneous testing dataset the model predicited with an accuracy of 97%.
- On the foreign dataset the model predicted with an accuracy of 60%. 

## Requirements

- Python 3.x
- numpy
- pandas
- scikit-learn
- matplotlib
- Pillow (PIL) (for image processing)

## Notes

- Ensure that the MNIST dataset (`train.csv`) is located in the same directory as `MNIST_KNN.py`.
- Personal handwritten images should be stored in the `images_of_numbers` folder in JPG format.

## Author

Nosipho Donkrag 

