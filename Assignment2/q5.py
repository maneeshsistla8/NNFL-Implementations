import numpy as np
import scipy.io

def sigmoid(s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

def tanh(x):
    return np.tanh(x)

class NeuralNetwork(object):
    def __init__(self, sizes):
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = {}
        self.activations = {}
        self.biases = {}
        
        # Weight Initialization
        for i in range(1, self.num_layers):
            self.weights[i] = np.random.randn(self.sizes[i-1], self.sizes[i])
            
        # Bias Initialization
        for i in range(1, self.num_layers):
            self.biases[i] = np.random.randn(self.sizes[i], 1)
        
        # Activations Initialization
        for i in range(1, self.num_layers):
            self.activations[i] = np.zeros([self.sizes[i], 1])
        
    def forward(self, X):
        
        self.activations[0] = X
        
        for i in range(1, self.num_layers):
            self.activations[i] = sigmoid(np.dot(self.weights[i].T, self.activations[i-1]) + self.biases[i])

        return self.activations[self.num_layers-1] 
    
    def backward(self, X, Y, output):
        
        self.delta = {}
        self.delta_output = (Y - output)*sigmoid(output, deriv=True)
        self.delta[self.num_layers-1] = self.delta_output
        
        # Delta caluclation
        for i in range(self.num_layers-1, 1, -1):
            self.delta[i-1] = np.dot(self.weights[i], self.delta[i])*sigmoid(self.activations[i-1], deriv=True)
        
        # Weight updation
        for i in range(1, self.num_layers-1):
            self.weights[i] += alpha*np.dot(self.activations[i-1], self.delta[i].T)
            
        # Bias updation
        for i in range(1, self.num_layers-1):
            self.biases[i] += alpha*self.delta[i]
        
    def train(self, X, Y):
        X = np.reshape(X, (len(X), 1))
        output = self.forward(X)
        self.backward(X, Y, output)
        
    def get_activations(self, x):
        x = np.reshape(x, (len(x), 1))
        self.forward(x)
        return self.activations
    
    def load_activations(self, activations):
        self.activations = activations
        
    def get_weights(self):
        return self.weights
    
    def load_weights(self, weights):
        self.weights = weights

def calculate_loss(NN,x ,y):
    
    loss = 0
    for i in range(len(x)):
        x_ = np.reshape(x[i], (len(x[i]), 1))
        loss += 0.5/len(x)*np.sum((y[i] - NN.forward(x_))**2)
    
    return loss


data = scipy.io.loadmat('data5.mat')
data = np.array(data['x'])
np.random.shuffle(data)
X = data[:,:-1]
labels = data[:, -1]

for i in range(X.shape[1]):
    X[:,i] = (X[:,i] - X[:,i].mean())/X[:,i].std()
    
X = (X - X.mean())/X.std()

y = np.zeros([len(X),2])

for i in range(len(labels)):
    if labels[i]==1:
        y[i,1] = 1.0
    elif labels[i]==0:
        y[i,0] = 1.0

x_train, y_train = X[:int(0.7*len(X))], y[:int(0.7*len(X))]
x_test, y_test = X[int(0.7*len(X)):], y[int(0.7*len(X)):]

alpha = 0.5

AE1 = NeuralNetwork([72, 60, 72])
AE2 = NeuralNetwork([60,50,60])

# Pretraining Autoencoder 1

for i in range(250):
    for j, row in enumerate(x_train):
        row = np.reshape(row, (72, 1))
        AE1.train(row, row)
        
        if j%500==0:
            loss = calculate_loss(AE1, x_train, x_train)
            print("Epoch {}, Loss {}".format(i, loss))

# Getting inputs for autoencoder 2
autoencoder2_input = []

for row in x_train:
    autoencoder2_input.append(AE1.get_activations(row)[1])
    
autoencoder2_input = np.array(autoencoder2_input)

# Pretraining autoencoder 2

for i in range(250):
    for j, row in enumerate(autoencoder2_input):
        row = np.reshape(row, (60,1))
        AE2.train(row, row)
        
        if j%500==0:
            loss = calculate_loss(AE2, autoencoder2_input, autoencoder2_input)
            print("Epoch {}, Loss {}".format(i, loss))

# Getting inputs for ELM
elm_input = []

for row in autoencoder2_input:
    elm_input.append(AE2.get_activations(row)[1])
    
elm_input = np.array(elm_input)

elm_neurons = 220
output_neurons = 2

W_elm = np.random.randn(elm_input.shape[1], elm_neurons)
b = np.random.randn(elm_neurons)

# ELM Training
elm_input = np.reshape(elm_input, (1503, 50))
H = np.matmul(elm_input, W_elm) + b
H = tanh(H)
H_inv = np.linalg.pinv(H)
W_final = np.matmul(H_inv, y_train)

# Testing

# Layer 1
layer1_out = []

for i, row in enumerate(x_test):
    act = AE1.get_activations(row)[1]
    layer1_out.append(act)
    
layer1_out = np.array(layer1_out)
layer1_out = np.reshape(layer1_out, (645, 60))

# Layer 2
layer2_out = []

for i, row in enumerate(layer1_out):
    act = AE2.get_activations(row)[1]
    layer2_out.append(act)
    
layer2_out = np.array(layer2_out)
layer2_out = np.reshape(layer2_out, (645, 50))

# ELM Layer
H_T = np.matmul(layer2_out, W_elm)
H_T = tanh(H_T)
y_pred = np.matmul(H_T, W_final)

TP,TN,FP,FN = 0,0,0,0

for i in range(len(y_pred)):

    if np.argmax(y_pred[i])==1 and np.argmax(y_test[i]) == 1:
        TP += 1
    elif np.argmax(y_pred[i])==0 and np.argmax(y_test[i]) == 0:
        TN += 1
    elif np.argmax(y_pred[i])==1 and np.argmax(y_test[i]) == 0:
        FP += 1
    elif np.argmax(y_pred[i])==0 and np.argmax(y_test[i]) == 1:
        FN += 1

accuracy = (TP+TN)/(TP+TN+FP+FN)
sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
print(accuracy, sensitivity, specificity)

print(TP,TN,FP,FN)



