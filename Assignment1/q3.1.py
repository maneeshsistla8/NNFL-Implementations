import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RidgeBGD(X, Y, theta, iters, alpha):
	costs = []
	weights = []
	m = len(Y)
	reg_parameter = 0.1

	for i in range(iters):
		prediction = np.dot(X, theta)
		error = prediction - Y
		M = np.multiply((1 - (alpha * reg_parameter * (1/m))), theta)
		theta = M - (alpha * (1/m) * X.T.dot(error))
		cost = (1/(2*m)) * (np.sum((X.dot(theta)-Y)**2) + (reg_parameter * np.dot(theta[1:], theta[1:].transpose())))
		costs.append(cost)
		weights.append(theta)

	return costs, weights

data = pd.read_excel('data.xlsx')
data.columns = ['x1', 'x2', 'y']
data = (data - data.mean())/data.std()

Y = np.array(data['y'])
X = data.loc[:,['x1', 'x2']]
X = np.c_[np.ones(X.shape[0]), X]
X = np.array(X)

theta = [0, 0, 0]
alpha = 0.01
iters = 1000

costs, weights = RidgeBGD(X, Y, theta, iters, alpha)
theta = weights[-1]
cost = costs[-1]
print(cost)
print("{:.4f}, {:.4f}, {:.4f}".format(theta[0], theta[1], theta[2]))

fig, ax = plt.subplots()
ax.plot(np.arange(iters), costs, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
plt.show()

W = np.asmatrix(weights)
weights_toplot = W[:, [1,2]]
costs_toplot = np.array(costs)
to_plot = np.c_[costs_toplot, weights_toplot]

z = np.array(to_plot[:,0]).flatten()
x = np.array(to_plot[:,1]).flatten()
y = np.array(to_plot[:,2]).flatten()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(x, y, z)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost')
plt.show()
	