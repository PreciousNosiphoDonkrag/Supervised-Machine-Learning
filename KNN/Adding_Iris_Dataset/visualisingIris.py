import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target



# Plot the data 
plt.figure(figsize=(8, 6)) 

# Scatter plot for Sepal Length vs Sepal Width
plt.scatter(X[:, 0], X[:, 1], c=y, s=80, label='Sepal', edgecolors='k')

# Scatter plot for Petal Length vs Petal Width
plt.scatter(X[:, 2], X[:, 3], c=y, s=80, label='Petal',  marker='s', edgecolors='k')

# Set labels and title
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.title('Iris Dataset - Sepal and Petal Characteristics')

# Add legend
plt.legend()

# Show the plot
plt.show() 

