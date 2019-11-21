import numpy as np
import pandas as pd
import matplotlib as plt
from itertools import product

import scipy.io
from sklearn.metrics import accuracy_score, confusion_matrix

def Gaussian_Membership(xsj, cij, sigmaij):
    # Gaussian Membership Function
    mew = np.exp(-0.5*(xsj - cij)**2/ sigmaij**2)
    return mew

def Membership_Layer(sample, cluster_means, cluster_std, num_classes):
    membership_dict = {} # alpha

    for j in range(sample.shape[0]):
        input_feature = sample[j]
        membership_values = []

        for i in range(num_classes):
            membership_values.append(Gaussian_Membership(input_feature, cluster_means[i][j], cluster_std[i][j]))

        membership_dict.update({j : membership_values})

    return membership_dict

def Power_Layer(membership_dict, cluster_means, cluster_std, pij):
    power_layer_dict = {}

    for j in range(len(membership_dict)):
        input_feature = membership_dict[j]
        power_layer_dict.update({j : input_feature**pij[:, j]})

    return power_layer_dict

def Fuzzification_Layer(power_layer_dict, num_classes, beta_indices):
    beta = []
    for rule in range(beta_indices.shape[0]):
        beta.append(np.prod([power_layer_dict[i][j] for i, j in enumerate(beta_indices[rule])]))

    beta = np.array(beta)
    return beta

def Weighted_Outputs(beta, weights):
    o_sk = beta.T.dot(weights)

    return o_sk

def Normalization_Layer(weighted_outputs):
    delta = np.sum(weighted_outputs)
    normalized_weighted_outputs = weighted_outputs/delta

    return normalized_weighted_outputs, delta

def dEdpij(xsj, cij, sigmaij, p, hsk, tsk, delta, weights, beta, membership):
    if np.isfinite(np.log(membership)) == False:
        return 0

    gradient = (((hsk - tsk)*((1 - hsk)/delta)).dot(weights.T)).dot(beta)*np.log(membership)

    return gradient

def dEdsigmaij(xsj, cij, sigmaij, p, hsk, tsk, delta, weights, beta):
    gradient = (((hsk - tsk)*((1 - hsk)/delta)).dot(weights.T)).dot(beta)*p*(xsj - cij)**2/sigmaij**3

    return gradient

def dEdw(hsk, tsk, delta, beta):
    gradient = (((hsk - tsk)*((1 - hsk)/delta))[:,np.newaxis].dot(beta[:,np.newaxis].T)).T

    return gradient

def dEdCij(xsj, cij, sigmaij, p, hsk, tsk, delta, weights, beta):
    gradient = (((hsk - tsk)*((1 - hsk)/delta)).dot(weights.T)).dot(beta)*p*(xsj - cij)/sigmaij**2

    return gradient

def train(X_train, y_train, weights, pij, iterations, cluster_means, cluster_std, num_classes, beta_indices, learning_rate):
    for iteration in range(iterations):
        gradient_cij = np.zeros(cluster_means.shape)
        gradient_sigmaij = np.zeros(cluster_std.shape)
        gradient_pij = np.zeros(pij.shape)
        gradient_wik = np.zeros(weights.shape)
        print(iteration)
        for s, sample in enumerate(X_train):
            membership_grade = Membership_Layer(sample, cluster_means, cluster_std, num_classes)
            modified_membership_grade = Power_Layer(membership_grade, cluster_means, cluster_std, pij)
            beta = Fuzzification_Layer(modified_membership_grade, num_classes, beta_indices)
            osk = Weighted_Outputs(beta, weights)
            hsk, delta = Normalization_Layer(osk)
            Cs = np.argmax(hsk)
            hs = np.zeros(num_classes)
            hs[Cs] = 1

            tsk = np.zeros(num_classes)
            tsk[np.int(y_train[s])] = 1

            for i in range(num_classes):
                for j in range(X_train.shape[1]):
                    xsj = sample[j]
                    cij = cluster_means[i][j]
                    sigmaij = cluster_std[i][j]

                    # Computing all the gradient
                    gradient_cij[i][j] += dEdCij(xsj, cij, sigmaij, pij[i][j], hsk, tsk, delta, weights, beta)
                    gradient_sigmaij[i][j] += dEdsigmaij(xsj, cij, sigmaij, pij[i][j], hsk, tsk, delta, weights, beta)
                    gradient_pij[i][j] += dEdpij(xsj, cij, sigmaij, pij[i][j], hsk, tsk, delta, weights, beta, membership_grade[j][i])

            gradient_wik += dEdw(hsk, tsk, delta, beta)

        cluster_means = cluster_means -  learning_rate*gradient_cij/X_train.shape[0]
        cluster_std = cluster_std - learning_rate*gradient_sigmaij/X_train.shape[0]
        pij = pij - 0.1*gradient_pij/X_train.shape[0]
        weights = weights - gradient_wik

    return weights, pij, cluster_means, cluster_std

def test(X_test, y_test, cluster_means, cluster_std, weights, num_classes, pij, beta_indices):
    y_pred = []

    for sample in X_test:
            membership_grade_test = Membership_Layer(sample, cluster_means, cluster_std, num_classes)
            modified_membership_grade_test = Power_Layer(membership_grade_test, cluster_means, cluster_std, pij)
            beta_test = Fuzzification_Layer(modified_membership_grade_test, num_classes, beta_indices)
            osk_test = Weighted_Outputs(beta_test, weights)
            hsk, delta = Normalization_Layer(osk_test)

            Cs = np.argmax(hsk)
            y_pred.append(Cs)

    cm = confusion_matrix(y_pred, y_test)
    accuracy = np.sum(np.diag(cm)/np.sum(cm))

    print(cm)
    print(accuracy)
    return cm, accuracy

np.random.seed(0)

dataset = pd.read_excel('data4.xlsx', header = None).sample(frac=1).reset_index(drop=True)
data = dataset.values

data[:, -1] = data[:, -1] - 1

# Spltting into training and testing datasets
split=int(np.round(0.7*data.shape[0]))
data_train = data[:split, :]
data_test = data[split:, :]

X_train = data_train[:, :-1]
y_train = data_train[:, -1]

X_test = data_test[:, :-1]
y_test = data_test[:, -1]

dataset.head()

num_classes = len(np.unique(y_train))

classes = [[] for i in range(num_classes)]

# Segregating samples into respective classes
for i, row in enumerate(data_train):
    label = np.int(row[-1])
    classes[label].append(row)

cluster_means = []
cluster_std = []

for cluster in classes:
    cluster_means.append(np.mean(np.array(cluster)[:, :-1], axis = 0))
    cluster_std.append(np.std(np.array(cluster)[:, :-1], axis = 0))

cluster_means = np.array(cluster_means)
cluster_std = np.array(cluster_std)

learning_rate = 0.1
iterations = 50

# Initialising pij, weights and indices for beta neurons
pij = 2*np.random.random((num_classes, X_train.shape[1]))
beta_indices = np.array(list(product(range(num_classes), repeat = X_train.shape[1])))
weights = np.random.random((beta_indices.shape[0], num_classes))

weights, pij, cluster_means, cluster_std = train(X_train, y_train, weights, pij, iterations, cluster_means, cluster_std, num_classes, beta_indices, learning_rate)

cm, accuracy = test(X_test, y_test, cluster_means, cluster_std, weights, num_classes, pij, beta_indices)



