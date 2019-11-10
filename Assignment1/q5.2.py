import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

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
X = np.array(data.loc[:,['x0', 'x1', 'x2']])
Y = np.array(data['y'])

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

w1 = [i[1] for i in weights]
w2 = [i[2] for i in weights]
fig2 = plt.figure()
ax = fig2.add_subplot(111,projection = '3d')
ww1 = np.linspace(min(w1)-0.2,max(w1)+0.2,100)
ww2 = np.linspace(min(w2)-0.2,max(w2)+0.2,100)
W = np.zeros([3,1])
J_cont = np.zeros([len(ww1),len(ww2)])

for i1,w_1 in enumerate(ww1):
	for i2,w_2 in enumerate(ww2):
		W[0] = weights[-1][0]
		W[1] = w_1
		W[2] = w_2
		h_x = np.dot(X,W)
		temp = np.array(h_x).ravel() - np.array(Y).ravel()
		J_cont[i1][i2] = (1/(2*len(Y))) * np.sum(temp**2)	

v = np.squeeze(costs)
www1,www2 = np.meshgrid(ww1,ww2)
www_1,www_2 = np.squeeze(ww1), np.squeeze(ww2)
plt.plot(x, y, z,'r-,',zorder = 10)
ax.plot_surface(www1,www2,J_cont,cmap=cm.coolwarm)
ax.set_xlabel('W1')
ax.set_ylabel('W2')
ax.set_zlabel('Cost function')
plt.show()