import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def LikelihoodRatioTest(test, train):
	acc, acc1, acc2, acc3 = 0, 0, 0, 0
	train_label1_features = train.iloc[:, :7].loc[train['label'] == 1]
	train_label2_features = train.iloc[:, :7].loc[train['label'] == 2]
	train_label3_features = train.iloc[:, :7].loc[train['label'] == 3]
	test_label1 = test.iloc[:, :7].loc[test['label'] == 1]
	test_label2 = test.iloc[:, :7].loc[test['label'] == 2]
	test_label3 = test.iloc[:, :7].loc[test['label'] == 3]
	prior_1 = train_label1_features.shape[0]/train.shape[0]
	prior_2 = train_label2_features.shape[0]/train.shape[0]
	prior_3 = train_label3_features.shape[0]/train.shape[0]
	cov_1 = np.cov(train_label1_features.T)
	cov_2 = np.cov(train_label2_features.T)
	cov_3 = np.cov(train_label3_features.T)
	X_1 = np.matrix(train_label1_features.T)
	X_2 = np.matrix(train_label2_features.T)
	X_3 = np.matrix(train_label3_features.T)
	mean_train1 = np.array(X_1.mean(1)).flatten()
	mean_train2 = np.array(X_2.mean(1)).flatten()
	mean_train3 = np.array(X_3.mean(1)).flatten()
	tomul_1 = prior_1/(2 * math.pi * np.linalg.det(cov_1)**0.5)
	tomul_2 = prior_2/(2 * math.pi * np.linalg.det(cov_2)**0.5)
	tomul_3 = prior_3/(2 * math.pi * np.linalg.det(cov_3)**0.5)
	test_data_features = test.iloc[:, :7]

	for i in range(test.shape[0]):
		test_data_point = np.array(test_data_features.iloc[i, :])
		aposteriori_1 = tomul_1 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train1), np.linalg.inv(cov_1)),(test_data_point - mean_train1)))
		aposteriori_2 = tomul_2 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train2), np.linalg.inv(cov_2)),(test_data_point - mean_train2)))
		aposteriori_3 = tomul_3 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train3), np.linalg.inv(cov_3)),(test_data_point - mean_train3)))
		
		max_ap = max(aposteriori_1, aposteriori_2, aposteriori_3) 

		if(max_ap == aposteriori_1):
			class_label = 1
		elif(max_ap == aposteriori_2):
			class_label = 2
		else:
			class_label = 3

		if(class_label == test.iloc[i, 7]):
			acc += 1
			if(class_label == 1):
				acc1 += 1
			elif(class_label == 2):
				acc2 += 1
			else:
				acc3 += 1

	return acc/test.shape[0], acc1/test_label1.shape[0], acc2/test_label2.shape[0], acc3/test_label3.shape[0]
		
data = pd.read_excel('data4.xlsx')
data = pd.DataFrame(data)
to_norm = data.iloc[:, 1:-1]
data.iloc[:, 1:-1] = (to_norm - to_norm.mean())/to_norm.std()

train = data.sample(frac=0.7, random_state=random.randint(1,1000))
test = data.drop(train.index)
train.rename(columns={train.columns[-1]:'label'}, inplace=True)
test.rename(columns={test.columns[-1]:'label'}, inplace=True)

accuracy, accuracy_1, accuracy_2, accuracy_3 = LikelihoodRatioTest(test, train)
print("Overall accuracy is {}".format(accuracy))
print("Individual class accuracies are {}, {}, {}".format(accuracy_1, accuracy_2, accuracy_3))