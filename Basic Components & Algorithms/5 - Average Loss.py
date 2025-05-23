import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
class_targets = [0, 1, 1]

# Categorical Cross-Entropy Loss

loss = -np.log(softmax_outputs[
    range(len(softmax_outputs)), class_targets])
avg_loss = np.mean(loss)

print("Loss:", loss)
print("Average Loss:", avg_loss)