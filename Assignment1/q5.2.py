import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def LAR_SGD(data, theta, iters, alpha):
	costs = []
	weights = []
	m = len(data)
	reg_parameter = 0.1

	for i in range(iters):
		sum_weights = 0
		data = data.reindex(np.random.permutation(data.index))
		X = np.array(data.loc[:,['x0', 'x1', 'x2']])
		Y = np.array(data['y'])
		prediction = np.dot(X[0], theta)
		error = prediction - Y[0]
		M = np.multiply(reg_parameter * (1/2), np.sign(theta))
		theta = theta - (alpha * (X[0].dot(error) + M))

		for weight in theta[1:]:
			sum_weights += abs(weight)

		cost = (1/(2*m)) * (np.sum((X.dot(theta)-Y)**2) + (reg_parameter * sum_weights))
		costs.append(cost)
		weights.append(theta)

	return costs, weights

data = pd.read_excel('data.xlsx')
data = (data - data.mean())/data.std()
data = np.c_[np.ones(data.shape[0]), data]
data = pd.DataFrame(data)
data.columns = ['x0', 'x1', 'x2', 'y']

theta = [0, 0, 0]
alpha = 0.01
iters = 1000

costs, weights = LAR_SGD(data, theta, iters, alpha)
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