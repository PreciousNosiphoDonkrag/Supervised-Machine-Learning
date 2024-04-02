#Just printing out some images from the mnist dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Read the MNIST dataset from the CSV file
current_dir = os.path.dirname(os.path.realpath(__file__))
file_path = os.path.join(current_dir, "train.csv")
df = pd.read_csv(file_path)
print(df.columns)
# Extract features (pixel values) and labels from the dataset
MNIST_data = df.values
X_data = MNIST_data[:, 1:]  # All the data points starting from the 2nd col (pixel values)
Y_data = MNIST_data[:, 0]   # All the rows of the 1st col (digits)

# Define a function to display images
def display_images(images, labels, num_images):
    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        plt.subplot(2, 5, i+1)
        
        #the images are flattened out into a [1x784] matrix. 
        #reshape it into a 28X28 matrix and apply a gray scale
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Display 10 images from the dataset
num_images_to_display = 10
display_images(X_data, Y_data, num_images_to_display)
