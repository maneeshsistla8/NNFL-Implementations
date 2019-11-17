import numpy as np
import pandas as pd
import random
import scipy.io as sio

def sigmoid(x):
	return 1/(1+np.exp(-x))

def sigmoid_der(x):
	return x*(1-x)

def MLP(train, test, iters, alpha, hidden1_size, hidden2_size):
	m = train.shape[0]
	X = np.array(train.iloc[:, :-1])
	Y = np.array(train.iloc[:, -1])
	W1 = np.random.randn(hidden1_size, X.shape[1])
	b1 = np.zeros((hidden1_size, 1))

	W2 = np.random.randn(hidden2_size, hidden1_size)
	b2 = np.zeros((hidden2_size, 1))
	W3 = np.random.randn(1, hidden2_size)
	b3 = np.zeros((1,1))

	loss = []

	for i in range(iters):
		Z1 = np.dot(W1, X.T) + b1
		A1 = sigmoid(Z1)
		Z2 = np.dot(W2, A1) + b2
		A2 = sigmoid(Z2)
		Z3 = np.dot(W3, A2) + b3
		Y_tilda = sigmoid(Z3)
		error = (1/m) * (np.sum(np.power((Y_tilda-Y),2)))
		loss.append(error)
		# if(i%100):
		# 	print(Y_tilda)

		delta_3 = (Y_tilda-Y)
		delta_2 = W3.T.dot(delta_3)*sigmoid_der(A2)
		delta_1 = W2.T.dot(delta_2)*sigmoid_der(A1)
		
		W3 = W3 - alpha * np.dot(delta_3, A2.T)
		b3 = b3 - alpha * np.sum(delta_3, axis=1, keepdims=True)
		W2 = W2 - alpha * np.dot(delta_2, A1.T)
		b2 = b2 - alpha * np.sum(delta_2, axis=1, keepdims=True)
		W1 = W1 - alpha * np.dot(delta_1, X)
		b1 = b1 - alpha * np.sum(delta_1, axis=1, keepdims=True)

	test_X = np.array(test.iloc[:, :-1])
	test_Y = np.array(test.iloc[:, -1])
	test_size = test.shape[0]

	Z1 = np.dot(W1, test_X.T) + b1
	A1 = sigmoid(Z1)
	Z2 = np.dot(W2,A1) + b2
	A2 = sigmoid(Z2)
	Z3 = np.dot(W3,A2) + b3
	Y_tilda = sigmoid(Z3)

	pred_labels = []

	for value in Y_tilda[0]:
		if value < 0.5:
			value = 0
		else:
			value = 1
		pred_labels.append(value)

	correct = 0

	for idx in range(test_size):
		if(pred_labels[idx] == test_Y[idx]):
			correct += 1

	accuracy = correct/test_size
	# print(pred_labels)
	return accuracy

test = sio.loadmat('data5.mat')
data = test['x']
data = pd.DataFrame(data)
to_norm = data.iloc[:, :-1]
data.iloc[:, :-1] = (to_norm - to_norm.mean())/to_norm.std()
train = data.sample(frac=0.7, random_state=random.randint(1,1000))
test = data.drop(train.index)

hidden1_size = 14
hidden2_size = 16
iters = 5000
alpha = 0.01

accuracy = MLP(train, test, iters, alpha, hidden1_size, hidden2_size)
print("Accuracy with train-test split of 70-30 is {}".format(accuracy))

data = data.sample(frac=1, random_state=random.randint(1,1000))
sz = int(len(data) * 0.2)
sum = 0

for i in range(5):
	start = i*sz
	end = (i+1)*sz
	if(i == 4):
		end = len(data)
	train = data.iloc[start:end, :]
	test = data.drop(train.index)
	accuracy = MLP(train, test, iters, alpha, hidden1_size, hidden2_size)
	print("Accuracy with cross validation for fold {}: {}".format(i+1, accuracy))
	sum += accuracy

print("Average accuracy with 5-fold cross validation is {}".format(sum/5))