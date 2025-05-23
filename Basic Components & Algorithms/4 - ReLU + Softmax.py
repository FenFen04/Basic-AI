import numpy as np
import matplotlib.pyplot as plt

def spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))  # Feature matrix
    y = np.zeros(samples * classes, dtype=int)  # Labels

    for class_number in range(classes):
        indexes = range(samples * class_number, samples * (class_number + 1))
        r = np.linspace(0.0, 2, samples)  # Radius
        theta = np.linspace(class_number * 4, (class_number + 1) * 4, samples) + np.random.randn(samples) * 0.3  # Angle with noise
        X[indexes] = np.c_[r * np.sin(theta), r * np.cos(theta)]
        y[indexes] = class_number
    
    return X, y


class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax:

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # For numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

X, y = spiral_data(samples=100, classes=3)

# dense layer
dense1 = Layer_Dense(2, 4)
dense2 = Layer_Dense(4, 3)

# activation functions
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

# forward pass
dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

output = activation2.output[:5]
print(output)


