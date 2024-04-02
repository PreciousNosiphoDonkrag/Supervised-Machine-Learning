#this file is dedicated to preparing the user's handwritten image for
#the knn algorithm. 
#The image must meet the following requirements to ensure data consistency
    #Grayscale (the MNIST images are all in gray)
    #Size: 28x28
    #ensure that the handwritten number is clearly visible
    #with a constransting background (black pen white page)
    #scale the pixels to the range [0,1]

import numpy as np
from PIL import Image  # this is a python library used for handling images
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def prepare_input_image(path):
     # Fetch the image and convert it to grayscale
    input_image = Image.open(path).convert("L")
    #plot_image(input_image, prepared = False)
    # Ensure the image has a grey mode
    if input_image.mode != "L":
        input_image = input_image.convert("L")
    
    # Resize the image to 28x28
    sized_img = input_image.resize((28, 28))
    
    #plot_image(sized_img, prepared = True)
    # Convert the image to an array
    array_img = np.array(sized_img, dtype='float32')
    
    # Use MinMaxScaler for normalization (this helped improve the accuracy score) 
    # pixels are automatically scaled consistently with the data
    scaler = MinMaxScaler()
    scaled_img = scaler.fit_transform(array_img)
    
    return scaled_img
    


 # Visualization of input images
def plot_image(sized_img, prepared = False):
    if(prepared): plt.title("Input Image after preperation")
    else: plt.title("Input Image before preperation")

    plt.imshow(sized_img, cmap='gray')
    plt.axis('off')
    plt.show()
    





