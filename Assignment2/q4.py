import numpy as np
import pandas as pd
import random
import scipy.io as sio

def Tanh(X, a, b):
	H = X.dot(a) + b
	H = np.tanh(H)

	return H

def Gaussian(X, a, b):
	H = np.zeros((X.shape[0], a.shape[1]))

	for i in range(X.shape[0]):
		for j in range(a.shape[1]):
			H[i][j] = np.exp(-1 * b[j] * (np.linalg.norm(X[i] - a[:,j])))

	return H

def train_ELM(train, n, act="G"):
	X = np.array(train.iloc[:, :-1])
	Y = np.array(train.iloc[:, -1])
	y = np.zeros((X.shape[0], 2))

	for i in range(Y.shape[0]):
		if Y[i] == 0:
			y[i] = [1, 0]
		else:
			y[i] = [0, 1]

	if act is "G":
		a = np.random.rand(X.shape[1], n)
		b = np.random.rand(n)
		H = Gaussian(X, a, b)
	elif act is "T":
		a = np.random.randn(X.shape[1], n)
		b = np.random.randn(n)
		H = Tanh(X, a, b)

	W = (np.linalg.pinv(H)).dot(y)

	return W, a, b

def test_ELM(test, W, n, a, b, act="G"):
	X = np.array(test.iloc[:, :-1])

	if act is "G":
		H = Gaussian(X, a, b)
	elif act is "T":
		H = Tanh(X, a, b)

	pred_labels = H.dot(W)
	print(pred_labels)
	return pred_labels

def Accuracy(test, pred_labels):
	Y = np.array(test.iloc[:, -1])

	acc_ovr, acc1, acc2 = 0, 0, 0
	m1, m2 = 0, 0
	for i in range(Y.shape[0]):
		if(Y[i] == 0):
			m1 += 1
			if(np.argmax(pred_labels[i]) == Y[i]):
				acc_ovr += 1
				acc1 += 1
		else:
			m2 += 1
			if(np.argmax(pred_labels[i]) == Y[i]):
				acc_ovr += 1
				acc2 += 1
        
	return acc_ovr/Y.shape[0], acc1/m1, acc2/m2

test = sio.loadmat('data5.mat')
data = test['x']
data = pd.DataFrame(data)
to_norm = data.iloc[:, :-1]
data.iloc[:, :-1] = (to_norm - to_norm.mean())/to_norm.std()
data = data.sample(frac=1, random_state=random.randint(1,1000))
sz = int(len(data) * 0.2)
acc_sum, acc1_sum, acc2_sum = 0, 0, 0

n = 220
act = "G"

for i in range(5):
	start = i*sz
	end = (i+1)*sz
	if(i == 4):
		end = len(data)
	train = data.iloc[start:end, :]
	test = data.drop(train.index)
	W, a, b = train_ELM(train, n, act)
	pred_labels = test_ELM(test, W, n, a, b, act)
	acc_ovr, acc1, acc2 = Accuracy(test, pred_labels)
	print("Individual class accuracies for fold {} : {}, {}".format(i+1, acc1, acc2))
	print("Overall accuracy for fold {} : {}".format(i+1, acc_ovr))
	acc_sum += acc_ovr
	acc1_sum += acc1
	acc2_sum += acc2

print("Average class accuracies with 5-fold cross validation are {}, {}".format(acc1_sum/5, acc2_sum/5))
print("Overall accuracy with 5-fold cross validation is {}".format(acc_sum/5))