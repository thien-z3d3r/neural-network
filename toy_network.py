import numpy as np

# Columns 0 and 1 represent XOR inputs; column 2 is a constant bias feature.
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

alpha, hidden_dim = 0.5, 4
np.random.seed(1)

synapse_0 = 2 * np.random.random((3, hidden_dim)) - 1
synapse_1 = 2 * np.random.random((hidden_dim, 1)) - 1

for j in range(60000):
    layer_1 = 1 / (1 + np.exp(-(np.dot(X, synapse_0))))
    layer_2 = 1 / (1 + np.exp(-(np.dot(layer_1, synapse_1))))

    # Output gradient: error multiplied by sigmoid derivative
    layer_2_delta = (layer_2 - y) * (layer_2 * (1 - layer_2))
    # Backpropagated gradient for hidden layer
    layer_1_delta = layer_2_delta.dot(synapse_1.T) * (layer_1 * (1 - layer_1))

    # Gradient descent parameter updates
    synapse_1 -= alpha * layer_1.T.dot(layer_2_delta)
    synapse_0 -= alpha * X.T.dot(layer_1_delta)

print("Predictions after 60,000 iterations:")
print(np.round(layer_2, 4))
