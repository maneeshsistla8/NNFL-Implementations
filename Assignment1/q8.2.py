import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sigmoid(z):
	return 1/(1 + np.exp(-z))

def LogisticRegression(train, theta, iters, alpha, class_label):
	X = np.array(train.iloc[:, :-1])
	Y = np.array(train.iloc[:, -1])
	m = Y.shape[0]

	for i in range(len(Y)):
		Y[i] -= class_label
		if(Y[i] != 0):
			Y[i] = 1
			
	for i in range(iters):
		z = np.dot(X, theta)
		h = sigmoid(z)
		loss = ((-Y * np.log(h)) - ((1-Y) * np.log(1-h))) / m
		gradient = np.dot(X.T, (h-Y)) / m
		theta -= alpha * gradient

	return theta

def Accuracy(test, theta_1, theta_2, theta_3):
	accuracy, accuracy_1, accuracy_2, accuracy_3 = 0, 0, 0, 0
	m1, m2, m3 = 0, 0, 0

	X = np.array(test.iloc[:, :-1])
	Y = np.array(test.iloc[:, -1])
	z1 = np.dot(X, theta_1)
	z2 = np.dot(X, theta_2)
	z3 = np.dot(X, theta_3)
	h1 = sigmoid(z1)
	h2 = sigmoid(z2)
	h3 = sigmoid(z3)
	m = Y.shape[0]

	for i in range(m):
		if(Y[i] == 1):
			m1 += 1
		elif(Y[i] == 2):
			m2 += 1
		else:
			m3 += 1

	for index in range(m):
		c1, c2, c3 = 0, 0, 0
		a = 1 if h1[index] > 0.5 else 0
		b = 1 if h2[index] > 0.5 else 0
		c = 1 if h3[index] > 0.5 else 0
		c1 += (a == 0)
		c1 += (c == 1)
		c2 += (a == 1)
		c2 += (b == 0)
		c3 += (b == 1)
		c3 += (c == 0)
		max_class = max(c1, c2, c3)
		if(c1 == max_class):
			a = 1
		elif(c2 == max_class):
			a = 2
		elif(c3 == max_class):
			a = 3
		
		if(a == Y[index]):
			accuracy += 1
			if(a == 1):
				accuracy_1 += 1
			elif(a == 2):
				accuracy_2 += 1
			elif(a == 3):
				accuracy_3 += 1

	return (accuracy_1/m1, accuracy_2/m2, accuracy_3/m3, accuracy/m)


data = pd.read_excel('data4.xlsx')
data = np.c_[np.ones(data.shape[0]), data]
data = pd.DataFrame(data)

theta = [0, 0, 0, 0, 0, 0, 0, 0]
alpha = 0.01
iters = 10000

to_norm = data.iloc[:, 1:-1]
data.iloc[:, 1:-1] = (to_norm - to_norm.mean())/to_norm.std()

train = data.sample(frac=0.6, random_state=random.randint(1,1000))
test = data.drop(train.index)

A, B, C = [1, 2], [2, 3], [1, 3]
train.rename(columns={train.columns[-1]:'label'}, inplace=True)
train_1 = train.loc[train['label'].isin(A)]
train_2 = train.loc[train['label'].isin(B)]
train_3 = train.loc[train['label'].isin(C)]

theta_1 = LogisticRegression(train, theta, iters, alpha, 1)
theta_2 = LogisticRegression(train, theta, iters, alpha, 2)
theta_3 = LogisticRegression(train, theta, iters, alpha, 3)

a1, a2, a3, a = Accuracy(test, theta_1, theta_2, theta_3)
print("One vs One individual class accuracies: {}, {}, {}".format(a1, a2, a3))
print("One vs One overall accuracy: {}".format(a))
