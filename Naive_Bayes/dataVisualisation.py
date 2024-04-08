#'plot_keywords'
#'movie_title'
#'imdb_score'
import numpy as np
from dataPreperation import start_here
import pandas as pd
import matplotlib.pyplot as plt
movies_df = start_here()
#print(movies_df.head())

#get the quartiles and min and max values
Q1 = np.percentile(movies_df['imdb_score'],25)
Q2 = np.percentile(movies_df['imdb_score'],50)
Q3 = np.percentile(movies_df['imdb_score'],75)
    
    
plt.figure(figsize=(8,6))
plt.boxplot(movies_df['imdb_score'], vert=False)
plt.title('IMDb scores distribution')
plt.xlabel("IMDb scores")  
plt.show()

# Count occurrences of each quality category
quality_counts = movies_df['quality'].value_counts()

# Extract category labels and counts
categories = quality_counts.index #retun an index object containing the labels of each category
counts = quality_counts.values

# Create bar plot
plt.bar(categories, counts, color=['blue', 'green', 'purple'])

# Add labels and title
plt.xlabel('Quality Categories')
plt.ylabel('Counts')
plt.title('Distribution of Quality Categories')

# Show plot
plt.show()