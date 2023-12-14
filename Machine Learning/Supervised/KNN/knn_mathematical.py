#The Aim of this porject is to give a mathematical explanation of
#K-Nearest Neighbors 
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

#list of data points that will be called classes, these usually come
#from a text file

points = {"green": [[2,4], [1,3], [2,3], [3,2], [2,1], [3,4], [3,3], [1,2]],
            "red": [[5,6], [4,5], [4,6], [6,6], [5,4]]}
#classify the following point as eithe green or red
new_point = [5,5]

#calculate the euclidean distance between the point were trying to 
#classify and the already existing points

def distanceCaluclator(x,y):
    return np.sqrt(np.sum(  (np.array(x)-np.array(y))**2   )) #return an array of distances

#define a class that will implement KNN
class KNN:
    def __init__(self, k=30): #constructor
        self.k = k #self works like 'this. for c#'
        self.point = None
        
    def fit(self,points):
        self.points = points

    def predict(self, new_point):
        distances = []
        for category in self.points:
            for point in self.points[category]:
                distance = distanceCaluclator(point, new_point)
                distances.append([distance, category])
        
        categories = [category[1] for category in sorted(distances)[:self.k]]
        result = Counter(categories).most_common(1)[0][0]
    
        return result
    

clf = KNN()
clf.fit(points)
print(clf.predict(new_point))

#visualization
ax = plt.subplot()
ax.grid(True, color="#323232")
ax.set_facecolor("#000000")
ax.figure.set_facecolor("#121212")
ax.tick_params(axis="x", color="white")
ax.tick_params(axis="y", color="white")

for point in points['green']:
    ax.scatter(point[0], point[1], color="#02C04A", s=60)

for point in points['red']:
    ax.scatter(point[0], point[1], color="#FF0000", s=60)
    
new_class = clf.predict(new_point)
color = "#FF0000" if new_class == "red" else "#02C04A"
ax.scatter(new_point[0], new_point[1], color =color, marker = "*", s=200, zorder=100)

for point in points['green']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#02C04A", linestyle="--", linewidth="1")

for point in points['red']:
    ax.plot([new_point[0], point[0]], [new_point[1], point[1]], color="#FF0000", linestyle="--", linewidth="1")

plt.show()