import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('./MNIST_data/train.csv')

#view first 10 rows of data in tabular form with the help of panda
#this will produce 10 rows with 785 columns, first column is labels of numbers
#the rest of the columns are pixel values
#print(df.head(n=10))

MNIST_data = df.values

X_data = MNIST_data[:,1:] #All the data points starting from the 2nd col
Y_data = MNIST_data[:,0] #all the rows of the 1st col
#print(X_data[0, :])
#20% of the data will be used for testing hence we have to split it

split = int(0.8*X_data.shape[0])

#Allocate
X_train = X_data[:split, :] #every row up to the value split 
Y_train = Y_data[:split] #every row up to split
X_test = X_data[split:,:]#every row from split
Y_test = Y_data[split:]



def MNIST_data():
    # Scale the pixel values to the range [0, 1]
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    return X_train_scaled, Y_train, X_test_scaled, Y_test
