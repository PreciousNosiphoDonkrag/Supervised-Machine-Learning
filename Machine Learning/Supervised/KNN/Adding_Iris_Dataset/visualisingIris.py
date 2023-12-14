import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Display the Iris dataset in tabular form
iris_df = pd.DataFrame(data=X, columns=iris.feature_names)
iris_df['target'] = y
print("Iris Dataset (First 5 rows):")
print(iris_df.head())

# Plot the data
plt.figure(figsize=(8, 6))

# Scatter plot for Sepal Length vs Sepal Width
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k', s=80, label='Sepal')

# Scatter plot for Petal Length vs Petal Width
plt.scatter(X[:, 2], X[:, 3], c=y, cmap='viridis', marker='s', edgecolors='k', s=80, label='Petal')

# Set labels and title
plt.xlabel('Length (cm)')
plt.ylabel('Width (cm)')
plt.title('Iris Dataset - Sepal and Petal Characteristics')

# Add legend
plt.legend()

# Show the plot
plt.show() 
