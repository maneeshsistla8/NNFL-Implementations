import numpy as np
import pandas as pd
import random
import scipy.io as sio

def compute_distance(datapoint, centroids):
    return np.sum(np.power(datapoint - centroids, 2))

def Gaussian(x, centroid, beta):
	return np.exp(-1 * beta * (np.linalg.norm(x - centroid)) ** 2)

def calculate_parameters(X, centroids, clusters):
	m = np.zeros(centroids.shape[0])
	norm_sum = np.zeros(centroids.shape[0])
	sigma = np.zeros(centroids.shape[0])
	beta = np.zeros(centroids.shape[0])

	for i in clusters:
		m[i] += 1

	for row in range(X.shape[0]):
		j = clusters[row]
		norm_sum[j] += np.linalg.norm(X[row] - centroids[j])

	for k in range(centroids.shape[0]):
		s = norm_sum[k]/m[k]
		print(m[k])
		if(m[k] < 2):
			sigma[k] = np.amin(sigma)
		else:
			sigma[k] = s
		beta[k] = 1/(sigma[k] ** 2)

	return beta

def KMeansClustering(X, k):
	a = np.random.randint(0, X.shape[0], k)
	centroids = X[a]
	prev_centroids = np.zeros(shape = centroids.shape)
	iters = 1000
	distances = np.zeros	((X.shape[0], k))
	clusters = np.zeros(X.shape[0])
	# print(X)

	for i in range(iters):
		for j in range(k):
			for row in range(X.shape[0]):
				distances[row][j] = compute_distance(X[row], centroids[j])

		clusters = np.argmin(distances, axis=1)

		point_sum = np.zeros((k, X.shape[1]))
		count = np.zeros(k)

		for row in range(X.shape[0]):
			c = clusters[row]
			point_sum[c] += X[row]
			count[c] += 1

		for j in range(k):
			if(count[j] == 0):
				centroids[j] = np.copy(prev_centroids[j])
			else:
				centroids[j] = point_sum[j]/count[j]

		if(np.amax(prev_centroids-centroids) < 1e-6):
			break

		prev_centroids = np.copy(centroids)

	return centroids, clusters

def Accuracy(test, pred_labels):
	Y = np.array(test.iloc[:, -1])
	acc_ovr, acc1, acc2 = 0, 0, 0
	m1, m2 = 0, 0
	for i in range(Y.shape[0]):
		if(Y[i] == 0):
			m1 += 1
			if(Y[i] == pred_labels[i]):
				acc_ovr += 1
				acc1 += 1
		else:
			m2 += 1
			if(Y[i] == pred_labels[i]):
				acc_ovr += 1
				acc2 += 1
        
	return acc_ovr/Y.shape[0], acc1/m1, acc2/m2

def trainRBFNN(train, k):
	train_X = np.array(train.iloc[:, :-1])
	train_Y = np.array(train.iloc[:, -1])
	centroids, clusters = KMeansClustering(train_X, k)
	beta = calculate_parameters(train_X, centroids, clusters)

	H = np.zeros((train_X.shape[0], centroids.shape[0]))
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			H[i, j] = Gaussian(train_X[i], centroids[j], beta[j])

	W = (np.linalg.pinv(H)) @ train_Y

	return W, centroids, beta

def testRBFNN(test, centroids, W, beta):
	test_X = np.array(test.iloc[:, :-1])
	test_Y = np.array(test.iloc[:, -1])
	H = np.zeros((test_X.shape[0], centroids.shape[0]))
	for i in range(H.shape[0]):
		for j in range(H.shape[1]):
			H[i, j] = Gaussian(test_X[i], centroids[j], beta[j])

	pred_labels = H @ W

	for i in range(test_X.shape[0]):
		if(pred_labels[i] > 0.5):
			pred_labels[i] = 1
		else:
			pred_labels[i] = 0

	return pred_labels

test = sio.loadmat('data5.mat')
data = test['x']
data = pd.DataFrame(data)
to_norm = data.iloc[:, :-1]
data.iloc[:, :-1] = (to_norm - to_norm.mean())/to_norm.std()
train = data.sample(frac=0.7, random_state=random.randint(1,1000))
test = data.drop(train.index)

k = 6

W, centroids, beta = trainRBFNN(train, k) 
pred_labels = testRBFNN(test, centroids, W, beta)

acc_ovr, acc1, acc2 = Accuracy(test, pred_labels)
print("Individual class accuracies for train-test split of 70-30 : {}, {}".format(acc1, acc2))
print("Overall accuracy : {}".format(acc_ovr))

data = data.sample(frac=1, random_state=random.randint(1,1000))
sz = int(len(data) * 0.2)
acc_sum, acc1_sum, acc2_sum = 0, 0, 0

for i in range(5):
	start = i*sz
	end = (i+1)*sz
	if(i == 4):
		end = len(data)
	train = data.iloc[start:end, :]
	test = data.drop(train.index)
	W, centroids, beta = trainRBFNN(train, k)
	pred_labels = testRBFNN(test, centroids, W, beta)

	acc_ovr, acc1, acc2 = Accuracy(test, pred_labels)
	print("Individual class accuracies for fold {} : {}, {}".format(i+1, acc1, acc2))
	print("Overall accuracy for fold {} : {}".format(i+1, acc_ovr))
	acc_sum += acc_ovr
	acc1_sum += acc1
	acc2_sum += acc2

print("Average class accuracies with 5-fold cross validation are {}, {}".format(acc1_sum/5, acc2_sum/5))
print("Overall accuracy with 5-fold cross validation is {}".format(acc_sum/5))