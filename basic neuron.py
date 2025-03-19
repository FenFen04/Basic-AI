import numpy as np
import matplotlib.pyplot as plt

inputs = [1, 2, 3, 4]
weights = [[0.2, 0.3, 0.5, -0.9], 
           [0.5, 0.92, -0.1, 0.89],
           [0.3, -0.2, -0.98, 0.7]]

biases = [2, 4, 6]
layer_output = []

for neuron_weights, neuron_biases in zip(weights, biases):
    print("Weight No. :" + str(neuron_weights) + "\nBias No. :" + str(neuron_biases))
    neuron_output = 0

    for neuron_input, weights in zip(inputs, neuron_weights):
        print("Input Num. : " + str(neuron_input) + "\nWeight Num. :" + str(weights))
        neuron_output += neuron_input*weights

    layer_output.append(neuron_output)
print("\nResults: ")
print(layer_output)
