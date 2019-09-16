import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def VectorisedLinearRegression(X, Y):
	X_T = np.transpose(X)
	tobe_inv = np.dot(X_T, X)
	inv = np.linalg.inv(tobe_inv)
	A = np.dot(inv, X_T)
	weights = np.dot(A, Y)

	return weights

data = pd.read_excel('data.xlsx')
data = (data - data.mean())/data.std()
data = np.c_[np.ones(data.shape[0]), data]
data = pd.DataFrame(data)
data.columns = ['x0', 'x1', 'x2', 'y']
X = np.array(data.loc[:,['x0', 'x1', 'x2']])
Y = np.array(data['y'])

weights = VectorisedLinearRegression(X, Y)

print(weights)