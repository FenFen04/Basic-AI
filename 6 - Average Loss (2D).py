import numpy as np

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 0]])

# prevent division by zero
pred_clipped = np.clip(softmax_outputs, 1e-7, 1 - 1e-7)

# check dimensionality
if len(class_targets.shape) == 1:
    confidences = pred_clipped[range(len(pred_clipped), class_targets)]

elif len(class_targets.shape) == 2:
    confidences = np.sum(pred_clipped * class_targets, axis=1)


predicted_confidence = -np.log(confidences)
print("Predicted Confidence:", predicted_confidence)