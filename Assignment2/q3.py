import numpy as np
import scipy.io as sio


def sigmoid(s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

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
        
    def feedforward(self, X):
        
        self.activations[0] = X
        
        for i in range(1, self.num_layers):
            self.activations[i] = sigmoid(np.dot(self.weights[i].T, self.activations[i-1]) + self.biases[i])

        return self.activations[self.num_layers-1] 
    
    def backprop(self, X, Y, output):
        
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
        output = self.feedforward(X)
        self.backprop(X, Y, output)
        
    def get_activations(self, x):
        x = np.reshape(x, (len(x), 1))
        self.feedforward(x)
        return self.activations
    
    def load_activations(self, activations):
        self.activations = activations
        
    def get_weights(self):
        return self.weights
    
    def load_weights(self, weights):
        self.weights = weights

def calculate_loss(NN, x ,y):
    
    loss = 0
    for i in range(len(x)):
        x_ = np.reshape(x[i], (len(x[i]), 1))
        # loss += 0.5/len(x)*np.sum((y[i] -feedforward(x_))**2)
    
    return loss


data = sio.loadmat('data5.mat')
data = np.array(data['x'])
np.random.shuffle(data)
X = data[:,:-1]
y = data[:, -1]
X = (X - X.mean())/X.std()

x_train, y_train = X[:int(0.7*len(X))], y[:int(0.7*len(X))]
x_test, y_test = X[int(0.7*len(X)):], y[int(0.7*len(X)):]

alpha = 0.5

AE1 = NeuralNetwork([72, 50, 72])
AE2 = NeuralNetwork([50,30,50])
AE3 = NeuralNetwork([30, 20, 30])
NN = NeuralNetwork([72,50,30,20, 1])

# Pretraining Autoencoder 1

for i in range(150):
    for j, row in enumerate(x_train):
        row = np.reshape(row, (72,1))
        AE1.train(row, row)
        
        if j%500==0:
            loss = calculate_loss(AE1, x_train, x_train)
            # print("Epoch {}, Loss {}".format(i, loss))

autoencoder2_input = []

for row in x_train:
    autoencoder2_input.append(AE1.get_activations(row)[1])

autoencoder2_input = np.array(autoencoder2_input)

# Pretraining autoencoder 2

for i in range(150):
    for j, row in enumerate(autoencoder2_input):
        row = np.reshape(row, (50,1))
        AE2.train(row, row)
        
        if j%500==0:
            loss = calculate_loss(AE2, autoencoder2_input, autoencoder2_input)
            # print("Epoch {}, Loss {}".format(i, loss))


autoencoder3_input = []

for row in autoencoder2_input:
    autoencoder3_input.append(AE2.get_activations(row)[1])

autoencoder3_input = np.array(autoencoder3_input)

# Pretraining autoencoder 3

for i in range(150):
    for j, row in enumerate(autoencoder3_input):
        row = np.reshape(row, (30,1))
        AE3.train(row, row)
        
        if j%500==0:
            loss = calculate_loss(AE3, autoencoder3_input, autoencoder3_input)
            # print("Epoch {}, Loss {}".format(i, loss))

# Get and load weights for final neural network

weights1 = AE1.get_weights()[1]
weights2 = AE2.get_weights()[1]
weights3 = AE3.get_weights()[1]

weights_final = {}
weights_final[1] = weights1
weights_final[2] = weights2
weights_final[3] = weights3
weights_final[4] = np.random.randn(20, 1)

NN.load_weights(weights_final)

# Training
for i in range(1000):
    # print("Epoch: ", i)
    for j in range(len(x_train)):
        NN.train(x_train[j], y_train[j])

accuracy, a1, a2, m1, m2 = 0, 0, 0, 0, 0
for i in range(len(x_test)):
	x = np.reshape(x_test[i], (len(x_test[i]), 1))
	x =feedforward(x)
	p = 0 if x[0]<0.5 else 1

	if p is 0:
		m1 += 1
		if p==y_test[i]:
			accuracy += 1
			a1 += 1
	else:
		m2 += 1
		if p==y_test[i]:
			accuracy += 1
			a2 += 1

accuracy = accuracy/len(x_test)
a1 = a1/m1
a2 = a2/m2

print("Individual class accuracies: {}, {}".format(a1, a2))
print("Overall accuracy: {}".format(accuracy))
