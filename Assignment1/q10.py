import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def LikelihoodRatioTest(test, train_label1_features, train_label2_features, cov_1, cov_2, prior_1, prior_2):
	X_1 = np.matrix(train_label1_features.T)
	X_2 = np.matrix(train_label2_features.T)
	mean_train1 = np.array(X_1.mean(1)).flatten()
	mean_train2 = np.array(X_2.mean(1)).flatten()
	tomul_1 = 1/(2 * math.pi * np.linalg.det(cov_1)**0.5)
	tomul_2 = 1/(2 * math.pi * np.linalg.det(cov_2)**0.5)
	test_data_features = test.iloc[:, :4]
	
	tp, fp, tn, fn = 0, 0, 0, 0, 0
	for i in range(test_data_features.shape[0]):
		test_data_point = np.array(test_data_features.iloc[i, :])
		likelihood_1 = tomul_1 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train1), np.linalg.inv(cov_1)),(test_data_point - mean_train1)))
		likelihood_2 = tomul_2 * np.exp(-0.5 * np.dot(np.dot((test_data_point - mean_train2), np.linalg.inv(cov_2)),(test_data_point - mean_train2)))
		likelihood_ratio = likelihood_1/likelihood_2
		prior_ratio = prior_2/prior_1
		if(likelihood_ratio > prior_ratio):
			if(test.iloc[i, 4] == 1):
				tp += 1
			else:
				fp += 1
		else:
			if(test.iloc[i, 4] == 2):
				tn += 1
			else:
				fn += 1

	return (tp+tn)/(tp+fp+tn+fn), tp/(tp+fn), tn/(tn+fp)
		

data = pd.read_excel('data3.xlsx')
data = pd.DataFrame(data)
to_norm = data.iloc[:, 1:-1]
data.iloc[:, 1:-1] = (to_norm - to_norm.mean())/to_norm.std()

train = data.sample(frac=0.6, random_state=random.randint(1,1000))
test = data.drop(train.index)
train.rename(columns={train.columns[-1]:'label'}, inplace=True)
test.rename(columns={test.columns[-1]:'label'}, inplace=True)

train_label1_features = train.iloc[:, :4].loc[train['label'] == 1]
train_label2_features = train.iloc[:, :4].loc[train['label'] == 2]

prior_1 = train_label1_features.shape[0]/train.shape[0]
prior_2 = train_label2_features.shape[0]/train.shape[0]

cov_1 = np.cov(train_label1_features.T)
cov_2 = np.cov(train_label2_features.T)

accuracy, sensitivity, specificity = LikelihoodRatioTest(test, train_label1_features, train_label2_features, cov_1, cov_2, prior_1, prior_2)
print("Accuracy is {}, Sensitivity is {}, Specificity is {}".format(accuracy, sensitivity, specificity))