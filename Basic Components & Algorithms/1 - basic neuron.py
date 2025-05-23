import numpy as np
import matplotlib.pyplot as plt

inputs = [1, 2, 3, 4, 6]
weights = [[0.2, 0.3, 0.5, -0.9, -0.8], 
           [0.5, 0.92, -0.1, 0.89, 0.69],
           [0.3, -0.2, -0.98, 0.7, -0.91]]

biases = [2, 4, 6]
layer_output = []
counter1 = 0
counter2 = 0

for neuron_weights, neuron_biases in zip(weights, biases):
    counter1 += 1
    print("----------------------------------")
    print("----------------------------------")
    print("Neuron: " + str(counter1)+ "\n")
    print("Weights: " + str(neuron_weights) + "\nBias No. :" + str(neuron_biases) + "\n")
    neuron_output = 0

    for neuron_input, weights in zip(inputs, neuron_weights):
        counter2 += 1
        print("----------------------------------")
        print("Calculation: " + str(counter2)+ "\n")
        print("Input Num. : " + str(neuron_input) + "\nWeight Num. :" + str(weights)+ "\n")
        neuron_output += neuron_input*weights

    layer_output.append(neuron_output)
print("\nResults: ")
print(layer_output)
