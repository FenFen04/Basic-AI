import numpy as np

inputs = [[1.0, 2.0, 3.0, 6.0],
          [4.0, 5.0, 6.0, 9.0],
          [7.0, 8.0, 9.0, 12.0]]

weights = [[-0.1, 0.2, 0.3, 0.69],
           [0.4, -0.5, 0.6, 0.12],
            [0.7, 0.8, -0.9, 0.3]]

biases = [2, 2.5, 3]

weights2 = [[0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]]
biases2 = [1, -2, 3]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
print(layer1_outputs)

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)