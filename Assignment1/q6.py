import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def distance(data, centroid):
    return np.sqrt(np.sum((data-centroid)**2))

def KMeansCluster(X, k):
	random_indices = np.random.choice(len(X), 2)
	centroids = X[random_indices]
	iters = 1000
	clusters = np.zeros(len(X))

	for it in range(iters):
		for i in range(len(X)):
			if distance(X[i], centroids[0]) < distance(X[i], centroids[1]):
				clusters[i] = 0
			else:
				clusters[i] = 1

		for i in range(k):
			pts = [X[j] for j in range(len(X)) if clusters[j] == i]
			centroids[i] = np.mean(pts, axis = 0)

	return centroids, clusters

data = pd.read_excel('data2.xlsx')
data.columns = ['x1', 'x2', 'x3', 'x4']
f1 = data['x1'].values
f2 = data['x2'].values
f3 = data['x3'].values
f4 = data['x4'].values
X = np.array(list(zip(f1, f2, f3, f4)))
k = 2

centroids, clusters = KMeansCluster(X, k)
print(centroids)

plt.figure()
plt.scatter(np.arange(len(X)), X[:, 0],c = clusters.flatten())
plt.title('FEATURE 1')
plt.xlabel('Index in dataset')
plt.ylabel('X1')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 1],c = clusters.flatten())
plt.title('FEATURE 2')
plt.xlabel('Index in dataset')
plt.ylabel('X1')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 2],c = clusters.flatten())
plt.title('FEATURE 3')
plt.xlabel('Index in dataset')
plt.ylabel('X1')
plt.show()
plt.figure()
plt.scatter(np.arange(len(X)), X[:, 3],c = clusters.flatten())
plt.title('FEATURE 4')
plt.xlabel('Index in dataset')
plt.ylabel('X1')
plt.show()