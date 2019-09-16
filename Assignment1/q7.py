import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def LogisticRegression(train, theta, iters, alpha):
	X = np.array(train.loc[:,['x0', 'x1', 'x2', 'x3', 'x4']])
	Y = np.array(train['y'])
	m = Y.shape[0]

	for i in range(len(Y)):
		Y[i] -= 1

	for i in range(iters):
		z = np.dot(X, theta)
		h = sigmoid(z)
		loss = ((-Y * np.log(h)) - ((1-Y) * np.log(1-h))) / m
		gradient = np.dot(X.T, (h-Y)) / m
		theta -= alpha * gradient

	return theta

def Accuracy(test, theta):
	tp, fp, tn, fn = 0, 0, 0, 0

	X = np.array(test.loc[:,['x0', 'x1', 'x2', 'x3', 'x4']])
	Y = np.array(test['y'])
	z = np.dot(X, theta)
	h = sigmoid(z)
	m = Y.shape[0]

	for i in range(len(Y)):
		Y[i] -= 1

	for index in range(m):
		a = 1 if h[index] > 0.5 else 0
		if(a == Y[index]):
			if(a == 0):
				tp += 1
			else:
				tn += 1
		else:
			if(a == 0):
				fp += 1
			else:
				fn += 1

	return (tp+tn)/(tp+fp+tn+fn), tp/(tp+fn), tn/(tn+fp)


data = pd.read_excel('data3.xlsx')
data = np.c_[np.ones(data.shape[0]), data]
data = pd.DataFrame(data)
data.columns = ['x0', 'x1', 'x2', 'x3', 'x4', 'y']

theta = [0, 0, 0, 0, 0]
alpha = 0.01
iters = 10000

to_norm = data.iloc[:, 1:-1]
data.iloc[:, 1:-1] = (to_norm - to_norm.mean())/to_norm.std()

train = data.sample(frac=0.6, random_state=random.randint(1,1000))
test = data.drop(train.index)
theta = LogisticRegression(train, theta, iters, alpha)

accuracy, sensitivity, specificity = Accuracy(test, theta)
print("Accuracy is {}, Sensitivity is {}, Specificity is {}".format(accuracy, sensitivity, specificity))