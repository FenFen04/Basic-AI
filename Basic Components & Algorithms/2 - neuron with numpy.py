import numpy as np

inputs = [1, 2, 3, 4, 6]
weights = [[0.2, 0.3, 0.5, -0.9, -0.8], 
           [0.5, 0.92, -0.1, 0.89, 0.69],
           [0.3, -0.2, -0.98, 0.7, -0.91]]

biases = [2.0, 4.0, 6.0]
output = []

output = np.dot(weights, inputs) + biases

print(output)           